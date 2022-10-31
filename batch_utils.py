import os
import subprocess
import fileinput


def write_var_to_file(config_file: str, variable_name: str, variable_content: str) -> None:
    from typing import TextIO, List
    urls: TextIO = open(config_file, 'r')
    lines: List[AnyStr] = urls.readlines()  # read all the lines of txt
    urls.close()

    for index, line in enumerate(lines):  # iterate over each line
        line_split: List[str] = line.split('=')
        var_name: str = line_split[0].strip()  # use strip() to remove empty space
        if var_name == variable_name:
            var_value: str = variable_content + "\n"
            line: str = F"{var_name} = {var_value}"

        lines[index] = line

    with open(config_file, 'w') as urls:
        urls.writelines(lines)  # save all the lines

    return