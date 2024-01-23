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
    sess = sys.argv[2] # so for example if something went wrong and we wanna to the experiment again, increase session number when starting

    subject_ID = int(re.findall(r'(?<=-)\d+', subject)[0])
    output_str = subject + '_' + sess
    output_dir = './output_data/'
    settings_file = './settings.yml'
    session_mode = int(sys.argv[3])

    if not os.path.exists('./output_data'):
        os.mkdir('./output_data')

    #if os.path.exists(output_dir):
    #    print("Warning: output directory already exists. Renaming to avoid overwriting.")
    #    output_dir = output_dir + datetime.now().strftime('%Y%m%d%H%M%S')
    
    # instantiate and run the session 
    experiment_session = ExperiSession(output_str, output_dir, settings_file, subject_ID, session_mode)
    experiment_session.run()


if __name__ == '__main__':
    main()
