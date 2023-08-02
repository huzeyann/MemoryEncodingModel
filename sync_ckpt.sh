#!/bin/bash

# Prompt the user for the directory path (with a default value of '/data/ckpt')
read -p "Enter the directory path [default: /data/ckpt]: " directory
directory=${directory:-"/data/ckpt"}

# Check if the directory exists
if [ ! -d "$directory" ]; then
  echo "Directory not found!"
  exit 1
fi

# Prompt the user for the remote machine details (with default values)
read -p "Enter the remote machine IP [default: 114.514.1919.810]: " remote_ip
remote_ip=${remote_ip:-"114.514.1919.810"}

read -p "Enter the remote machine username [default: huze]: " remote_username
remote_username=${remote_username:-"huze"}

read -p "Enter the remote machine destination directory [default: /data/huze/dckpt]: " remote_directory
remote_directory=${remote_directory:-"/data/huze/dckpt"}

# Check if the remote machine is accessible
ping -c 1 "$remote_ip" > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Remote machine is not accessible!"
  exit 1
fi

# Function to walk through the directory recursively
walk_directory() {
  local dir="$1"

  # Loop through all the files and subdirectories in the current directory
  for file in "$dir"/*; do
    # Check if the item is a directory
    if [ -d "$file" ]; then
      # Check if the directory contains a 'done' file
      if [ -f "$file/done" ]; then
        echo "Syncing folder: $file"
        # Rsync the folder to the remote machine
        rsync -avz --progress "$file" "$remote_username@$remote_ip:$remote_directory"
      fi
      # Call the function recursively for the subdirectory
      walk_directory "$file"
    fi
  done
}

# Call the function to walk through the directory
walk_directory "$directory"