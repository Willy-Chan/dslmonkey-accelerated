#!/usr/bin/env python3
"""
Script to split a PyTorch file into multiple KernelBench-formatted sections using Gemini-3-pro-preview.

Usage:
    python split_to_kernelbench.py <input_pytorch_file.py>

Requires:
    - GEMINI_API_KEY environment variable set
    - google-genai package installed (pip install google-genai)
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed. Run: pip install google-genai")
    sys.exit(1)


KERNELBENCH_OUTPUT_DIR = Path(__file__).parent.parent / "dsl-monkeys" / "KernelBench" / "KernelBench" / "level5"

SYSTEM_PROMPT = """You are an expert at decomposing PyTorch programs into smaller, self-contained kernel operations.

Your task is to take a PyTorch program and split it into 8-9 smaller sections, where each section:
1. Represents a single, well-defined computational kernel
2. Can be independently optimized/compiled
3. Follows the KernelBench format exactly

Each section MUST follow this exact format:

```python
# SECTION: <number>_<short_snake_case_name>
import torch
import torch.nn as nn

class Model(nn.Module):
    \"\"\"
    <Description of what this kernel does>
    \"\"\"
    def __init__(self, <params>):
        super(Model, self).__init__()
        # Store any needed parameters
        
    def forward(self, <inputs>) -> <output_type>:
        \"\"\"
        Args:
            <document each input with shape>
        Returns:
            <document output with shape>
        \"\"\"
        # Implementation
        return output

# Kernelbench Parameters
<define any constants like batch_size, seq_len, etc.>

def get_inputs():
    # Return a list of input tensors for testing
    return [<tensors>]

def get_init_inputs():
    # Return a list of arguments for Model.__init__
    return [<init_args>]
```

CRITICAL REQUIREMENTS:
1. Each section MUST start with exactly: # SECTION: <number>_<name>
2. Each section MUST be a complete, runnable Python file
3. Each section MUST have the Model class with __init__ and forward methods
4. Each section MUST have get_inputs() and get_init_inputs() functions
5. Sections should be numbered 1 through 8 or 9
6. Names should be descriptive snake_case (e.g., 1_chunked_intra_attn, 2_kv_state_update)
7. Each kernel should do ONE logical operation (e.g., one matmul pattern, one reduction, one elementwise op)
8. Preserve the mathematical correctness - the composition of all kernels should be equivalent to the original

Think about how to decompose the program into fundamental operations that could each benefit from custom Triton kernels."""


def build_prompt(source_code: str) -> str:
    return f"""Please decompose the following PyTorch program into 8-9 smaller KernelBench-formatted sections.

Each section should represent a single kernel operation that could be independently optimized.

SOURCE CODE:
```python
{source_code}
```

Output each section using the exact format specified, with each section starting with:
# SECTION: <number>_<name>

Make sure each section is a complete, self-contained Python file that can run independently."""


def extract_sections(response_text: str) -> list[tuple[str, str]]:
    """
    Extract sections from the Gemini response.
    Returns list of (filename, code) tuples.
    """
    sections = []
    
    # Pattern to match section headers and their code
    # Look for # SECTION: followed by the name, then capture until next section or end
    pattern = r'# SECTION:\s*(\d+_[a-zA-Z0-9_]+)\s*\n(.*?)(?=# SECTION:|$)'
    
    # First try to find sections within code blocks
    code_block_pattern = r'```python\s*\n(.*?)```'
    code_blocks = re.findall(code_block_pattern, response_text, re.DOTALL)
    
    if code_blocks:
        # Process each code block
        for block in code_blocks:
            section_match = re.match(r'# SECTION:\s*(\d+_[a-zA-Z0-9_]+)\s*\n', block)
            if section_match:
                name = section_match.group(1)
                # Remove the section header line from the code
                code = block[section_match.end():].strip()
                # But we want to keep imports, so actually keep everything after the header
                code = re.sub(r'^# SECTION:.*\n', '', block).strip()
                sections.append((name, code))
    else:
        # Fallback: try to find sections directly in text
        matches = re.findall(pattern, response_text, re.DOTALL)
        for name, code in matches:
            code = code.strip()
            # Remove markdown code block markers if present
            code = re.sub(r'^```python\s*\n?', '', code)
            code = re.sub(r'\n?```\s*$', '', code)
            sections.append((name.strip(), code.strip()))
    
    return sections


def call_gemini(source_code: str, api_key: str) -> str:
    """Call Gemini-3-pro-preview with the decomposition prompt."""
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=build_prompt(source_code),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=16000,
        ),
    )
    
    return response.text


def write_sections(sections: list[tuple[str, str]], output_dir: Path, dry_run: bool = False):
    """Write extracted sections to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    written_files = []
    for name, code in sections:
        filename = f"{name}.py"
        filepath = output_dir / filename
        
        if dry_run:
            print(f"\n{'='*60}")
            print(f"Would write: {filepath}")
            print(f"{'='*60}")
            print(code[:500] + "..." if len(code) > 500 else code)
        else:
            filepath.write_text(code)
            written_files.append(filepath)
            print(f"Written: {filepath}")
    
    return written_files


def main():
    parser = argparse.ArgumentParser(
        description="Split a PyTorch file into KernelBench-formatted sections using Gemini"
    )
    parser.add_argument("input_file", type=Path, help="Input PyTorch .py file to split")
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=KERNELBENCH_OUTPUT_DIR,
        help=f"Output directory for generated files (default: {KERNELBENCH_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print what would be written without actually writing files"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key (defaults to GEMINI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set and --api-key not provided")
        sys.exit(1)
    
    # Read input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    source_code = args.input_file.read_text()
    print(f"Read {len(source_code)} bytes from {args.input_file}")
    
    # Call Gemini
    print("Calling Gemini-3-pro-preview...")
    response_text = call_gemini(source_code, api_key)
    print(f"Received response ({len(response_text)} chars)")
    
    # Extract sections
    sections = extract_sections(response_text)
    print(f"Extracted {len(sections)} sections")
    
    if not sections:
        print("Error: No sections extracted from response. Raw response:")
        print(response_text[:2000])
        sys.exit(1)
    
    # Write files
    written = write_sections(sections, args.output_dir, dry_run=args.dry_run)
    
    if not args.dry_run:
        print(f"\nSuccessfully wrote {len(written)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
