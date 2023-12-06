import requests
from bs4 import BeautifulSoup
import os
import sys
import argparse
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

def main(config):
    url = 'https://elastic-cense.ls2n.fr/data/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    output_dir = config.output

    if not os.path.exists(output_dir):
        # Create the directory recursively
        os.makedirs(output_dir)

    urls = []
    for link in soup.find_all('a'):
        link_ref = link.get('href')
        if "cense_dump" in link_ref:
            data = link.get('href')

            to_run = "wget -P " + output_dir + " " + url + data 
            os.system(to_run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='Output directory for the datasets', default= './cense_data/')
    config = parser.parse_args()
    main(config)