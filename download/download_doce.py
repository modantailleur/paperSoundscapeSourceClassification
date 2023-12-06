import os
import shutil
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

# Define the name of the folder to clone
folder_name = "doce"

# Check if the folder exists
if os.path.exists(folder_name):
    # If the folder exists, delete it
    shutil.rmtree(folder_name)

# Define the command to execute
command = "git clone https://github.com/mathieulagrange/doce"

# Execute the command using os.system()
os.system(command)

# Move the child "doce" folder to the same level as the parent folder and rename it to "doce_temp"
shutil.move(os.path.join(folder_name, "doce"), "doce_temp")

# Delete the parent "doce" folder
shutil.rmtree(folder_name)

# Rename the "doce_temp" folder to "doce"
os.rename("doce_temp", folder_name)