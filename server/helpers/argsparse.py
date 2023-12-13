import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port")
parser.add_argument("-r", "--redis", action="store_true")

args = parser.parse_args()