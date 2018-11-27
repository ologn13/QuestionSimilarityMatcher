import re
from string import maketrans

def remove_html_tags(data):
    """
    This removes the html tags from the data
    """
    p = re.compile(r'<.*?>')
    return p.sub(' ', data)


def process_line(line=""):
    """
    Removes html tags, converts special characters to whitespace and transform every character to lower character.
    A list is then returned containing all the tokens splitted by whitespace.
    """
    if line is None:
        return ""
    try:
        line = str(line).decode('ascii', 'ignore')
        line = str(remove_html_tags(line)).translate(maketrans(
            "-=!()/\\0123456789.?%,\"*[];&@#:'+", "                                ")).lower()
    except Exception:
        raise Exception("Issue in line: " + line)
    return line.split()

