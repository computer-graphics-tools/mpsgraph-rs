#!/usr/bin/env python3
import os
import re
import glob

# Find all Rust files in the src directory
rust_files = glob.glob("src/**/*.rs", recursive=True)

# Regular expression to find msg_send! macro calls
msg_send_pattern = re.compile(r'msg_send!\[(.*?)\];', re.DOTALL)

# Regular expression to find parameters in msg_send! calls
param_pattern = re.compile(r'(\s+)([a-zA-Z0-9_]+):\s+([^\s,]+)(\s*?)$', re.MULTILINE)

# Process each Rust file
for filepath in rust_files:
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Find all msg_send! macro calls
    matches = msg_send_pattern.findall(content)
    
    if not matches:
        continue
    
    # Process each msg_send! call
    for match in matches:
        # If there's already a comma after each parameter, skip
        if re.search(r'(\s+)([a-zA-Z0-9_]+):\s+([^\s,]+)(\s*?)$', match, re.MULTILINE):
            # Replace parameters without commas with parameters with commas
            new_match = param_pattern.sub(r'\1\2: \3,\4', match)
            
            # Replace the original match with the new match in the content
            content = content.replace(match, new_match)
    
    # Write the updated content back to the file
    with open(filepath, 'w') as file:
        file.write(content)
    
    print(f"Processed {filepath}")

print("All files processed.") 