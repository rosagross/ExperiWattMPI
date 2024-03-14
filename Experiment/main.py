#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@time    :   2024/01/17 13:31:04
@author  :   rosagross
@contact :   grossmann.rc@gmail.com
'''


import sys
import os
import re
from datetime import datetime
from session import ExperiSession
datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    subject = sys.argv[1]

    subject_ID = int(re.findall(r'(?<=-)\d+', subject)[0])
    output_str = subject
    output_dir = './output_data/'
    settings_file = './settings.yml'
    session_mode = int(sys.argv[2])

    if not os.path.exists('./output_data'):
        os.mkdir('./output_data')

    # instantiate and run the session 
    experiment_session = ExperiSession(output_str, output_dir, settings_file, subject_ID, session_mode)
    experiment_session.run()


if __name__ == '__main__':
    main()
