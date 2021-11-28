"""
pp.py

Forum-posting NNs try to ID some NAOs.

Copyright 2021 Nathan Mifsud <nathan@mifsud.org>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
import sys
import nltk
import random
import datetime
import markovify
import numpy as np


def generate_username():
    adj = random.choice(['Fast','Faster','Retina','Mask','Efficient',
                         'YOLO','SSD','SPP','Refine','Cascade','Stair',
                         'Squeeze','FRD','Res','Scout','Look','Resolve',
                         'Search','Salient','Detect','Context','Mobile',
                         'Rapid','Speed','Vision','Eye','View','Seek'])
    add = random.choice(['','-',' ']) + \
          random.choice(['RCNN','CNN','Net','Det','NeXt','LITE','9000'])
    num = min(random.randint(2,50), random.randint(2,50))
    ver = random.choice(['-',' ']) + random.choice(['R','v','D']) + str(num)
    opt = random.choices([add,''],weights=[6,1]) + \
          random.choices([ver,''],weights=[4,1])
    username = adj + ''.join(opt)
    return username


def generate_users(num_users):
    users = {}
    existing_usernames = set()
    for _ in range(num_users):
        unique = False
        while not unique: # prevent duplication of usernames
            username = generate_username()
            if username not in existing_usernames:
                existing_usernames.add(username)
                break
        post_count = min(random.randint(1,7777), random.randint(1,7777),
                         random.randint(1,7777), random.randint(1,7777))
                     # weighted so relatively few users have high post counts
        users[username] = post_count
    return users


def user_rank(post_count):
    if           post_count <  100:  rank = 'Lone Neuron'
    elif 100  <= post_count <  250:  rank = 'Backprop Kid'
    elif 250  <= post_count <  500:  rank = 'Semi-Supervised'
    elif 500  <= post_count <  1000: rank = 'Unsupervised'
    elif 1000 <= post_count <  1500: rank = 'Fully Recurrent'
    elif 1500 <= post_count <  3000: rank = 'Neuroevolved'
    elif 3000 <= post_count <  5000: rank = 'Hyperoptimised'
    elif         post_count >= 5000: rank = 'Perfectly Parallel'
    return rank


def build_models():
    class POSifiedText(markovify.Text):
        def word_split(self, sentence):
            words = re.split(self.word_split_pattern, sentence)
            words = [ '::'.join(tag) for tag in nltk.pos_tag(words) ]
            return words
        def word_join(self, words):
            sentence = ' '.join(word.split('::')[0] for word in words)
            return sentence

    with open('corpus-science.txt') as f:
        science  = f.read() # papers on object detection
    with open('corpus-queries.txt') as f:
        queries = f.read() # first posts
    with open('corpus-replies.txt') as f:
        replies = f.read() # replies from other users
    with open('corpus-updates.txt') as f:
        updates = f.read() # updates and replies from OPs

    m_science = markovify.Text(science)
    m_queries = markovify.Text(queries)
    m_replies = markovify.Text(replies)
    m_updates = markovify.Text(updates)

    return m_science, m_queries, m_replies, m_updates


def build_thread(minimum_word_count):
    m_science, m_queries, m_replies, m_updates = build_models()
    posts, pages = [], []
    prev_quote, next_quote = '', ''
    posted_images = set()
    date = datetime.datetime(2021,11,9,8)
    users = generate_users(50)
    word_count = 0

    for post in range(1000):
        # determine which user posted
        if post == 0: user, last_user = 0, 0
        while user == last_user: # prevent double posting
            if post > 0 and np.random.rand() > .8:
                user = original_poster
            else:
                user = min(random.randint(0,49), random.randint(0,49))
        last_user = user
        username = list(users.keys())[user]
        post_count = users[username]

        # determine when they posted
        if post == 0: post_date = date
        interval = min(random.randint(5,240), random.randint(5,240))
        post_date += datetime.timedelta(minutes=interval)
        datestamp = post_date.strftime('%-d %B %Y')
        timestamp = post_date.strftime('%-d %b %Y at %-I:%M %p')

        # tag OP and define post type
        if post == 0:
            original_poster = user
            post_type = m_queries
            author = '' # don't need author tag on first post
        elif post > 0 and user == original_poster:
            post_type = m_updates
            author = '<span class="OP">Author</span>'
        elif user != original_poster:
            post_type = m_replies
            author = ''

        # bias model by post count so veteran users sound more technical
        model = markovify.combine([m_science, post_type], [post_count, 5000])

        # sometimes start the post with a quote
        if np.random.rand() > .9 and len(next_quote) > 0 \
            and next_quote != prev_quote:
            quote = '<blockquote>' + next_quote + '</blockquote>'
        else:
            quote = ''

        # generate new sentences split into paragraphs
        content = []
        num_para = 6 if post == 0 else random.randint(1,6)
        for p in range(num_para):
            para = '<p>'
            num_sentences = random.choice([1,1,1,2,2,2,3,3,4,4,5])
            for s in range(num_sentences):
                sentence = model.make_sentence()
                if isinstance(sentence, str):
                    para += sentence
                else: # too much overlap with the original text
                    print('make_sentence() returned None')
                if np.random.rand() > .95:
                    para += ' ' + random.choice(['😀','🤣','🙂','🧐','🤔',
                                                 '😲','😉','🙄','😕','😆',
                                                 '👀','🤖','🦕','🌿','🍄'])
                para += ' ' if s+1 < num_sentences else ''
            para += '</p>'

            # select certain paragraphs to be quoted later
            if np.random.rand() > .5:
                prev_quote = next_quote
                next_quote = '<p><b>On ' + timestamp + ', ' + username + \
                             ' said:</b></p>' + para

            # if OP, sometimes post yet-unseen images
            post_image = False
            if (post == 0 and p > 2) or np.random.rand() > .7:
                post_image = True
            if user == original_poster \
                and post_image == True \
                and len(posted_images) < 100: # avoid hanging if supply exhausted
                unique = False
                while not unique: # each image only appears once
                    image = random.randint(1,100)
                    if image not in posted_images:
                        posted_images.add(image)
                        break
                para += '<img src="/img/pp/object/' + str(image) + '.png" alt="">'
        
            content += para

        content = quote + ''.join(content)

        # format the post in HTML
        with open('template-post.html', 'r') as f:
            html = f.read() \
                    .replace('POST_ID', str(post+1)) \
                    .replace('USER_ID', str(user+1)) \
                    .replace('USERNAME', username) \
                    .replace('OP', author) \
                    .replace('DATETIME', timestamp) \
                    .replace('DATE', datestamp) \
                    .replace('USER_RANK', user_rank(post_count)) \
                    .replace('POST_COUNT', str(post_count)) \
                    .replace('POST_CONTENT', ''.join(content))

        posts += html

        # every 20 posts, start a new page
        if (post+1) % 20 == 0:
            pages.append(''.join(posts))
            posts = [] # reset for next loop

        # stop posting once the thread hits the word count
        word_count += len(content.split())
        if word_count > minimum_word_count:
            if (post+1) % 20 != 0: # append remaining posts
                pages.append(''.join(posts))
            total_posts = post+1
            total_pages = len(pages)
            break

    # generate the HTML files
    for p in range(total_pages):
        # this pagination code is crude but it works!
        hide, t1, t2, t3 = '', '<a href="/pp/', '"><li>', '</li></a>\n            '
        prev_4 = hide if p < 4             else t1 + str(p-3) + t2 + str(p-3) + t3
        prev_3 = hide if p < 3             else t1 + str(p-2) + t2 + str(p-2) + t3
        prev_2 = hide if p < 2             else t1 + str(p-1) + t2 + str(p-1) + t3
        prev_1 = hide if p < 1             else t1 + str(p)   + t2 + str(p)   + t3
        next_1 = hide if p+2 > total_pages else t1 + str(p+2) + t2 + str(p+2) + t3
        next_2 = hide if p+3 > total_pages else t1 + str(p+3) + t2 + str(p+3) + t3
        next_3 = hide if p+4 > total_pages else t1 + str(p+4) + t2 + str(p+4) + t3
        next_4 = hide if p+5 > total_pages else t1 + str(p+5) + t2 + str(p+5) + t3

        end_prev = hide if p < 1 else t1 + t2 + '«' + t3 \
                    + t1 + str(p) + t2 + 'Prev' + t3
        end_next = hide if p+2 > total_pages else t1 + str(p+2) + t2 + 'Next' + t3 \
                    + t1 + str(total_pages) + t2 + '»' + t3[:18]

        with open('template-page.html', 'r') as f:
            html = f.read() \
                    .replace('POSTS',        pages[p]) \
                    .replace('CURRENT_PAGE', str(p+1)) \
                    .replace('TOTAL_PAGES',  str(total_pages)) \
                    .replace('PREV_4',       prev_4) \
                    .replace('PREV_3',       prev_3) \
                    .replace('PREV_2',       prev_2) \
                    .replace('PREV_1',       prev_1) \
                    .replace('NEXT_1',       next_1) \
                    .replace('NEXT_2',       next_2) \
                    .replace('NEXT_3',       next_3) \
                    .replace('NEXT_4',       next_4) \
                    .replace('END_PREV',     end_prev) \
                    .replace('END_NEXT',     end_next) \
                    .replace('"/pp/1"', '"/pp/"') # go to index instead

        filename = 'index.html' if p == 0 else str(p+1) + '.html'
        with open(filename, 'w') as f:
            f.write(html)

    print(str(total_pages) + ' pages | ' \
        + str(total_posts) + ' posts | ' + str(word_count) + ' words')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        build_thread(50000)
    else: # generate thread with a specific word count
        build_thread(int(sys.argv[1]))