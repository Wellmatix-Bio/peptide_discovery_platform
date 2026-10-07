#!/bin/bash
current_branch=$(git symbolic-ref --short HEAD) # get the current branch name
echo "Current branch: $current_branch"
echo "Please enter the commit message:"
read name 
dt=$(date '+%d/%m/%Y %H:%M:%S');  
git add .
git commit -m "$name at $dt"
git push origin $current_branch 
sleep 3

echo "Pushed to $current_branch branch"
