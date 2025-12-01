#!/usr/bin/env python3
"""Fix Python 3.10+ union syntax to Python 3.8+ compatible Optional syntax."""

import sys

files_to_fix = [
    'sav_benchmark/runners/edgetam.py',
    'sav_benchmark/runners/sam2.py'
]

for filepath in files_to_fix:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Count occurrences before replacement
        str_none_count = content.count('str | None')
        float_none_count = content.count('float | None')
        int_none_count = content.count('int | None')
        
        # Replace all union syntax patterns
        content = content.replace('str | None', 'Optional[str]')
        content = content.replace('float | None', 'Optional[float]')
        content = content.replace('int | None', 'Optional[int]')
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Fixed {filepath}")
            print(f"  - str | None: {str_none_count} replacements")
            print(f"  - float | None: {float_none_count} replacements")
            print(f"  - int | None: {int_none_count} replacements")
        else:
            print(f"- No changes needed in {filepath}")
    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

print("\n✓ All files fixed successfully!")
