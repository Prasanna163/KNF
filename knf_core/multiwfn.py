import os
import subprocess
import logging
from .utils import run_subprocess

def create_multiwfn_script(script_path: str):
    """
    Creates a Multiwfn input script for NCI analysis.
    Sequence from plan: 20, 1, 3, 1, 2, 0, 0, q
    """
    # The sequence is:
    # 20: Visual study of IRI/RDG/IGM
    # 1: RDG (Reduced Density Gradient)
    # 3: ? (Likely grid quality or range setting)
    # 1: ?
    # 2: ?
    # 0: ?
    # 0: ?
    # q: Quit
    
    # We will write explicitly to a file
    # Note: Multiwfn reads from stdin, so we need newlines.
    
    commands = [
        "20", # Visual analysis
        "1",  # RDG
        "3",  # High quality grid? Or option 3?
        "1",
        "2",
        "0",
        "0",
        "q" 
    ]
    
    with open(script_path, 'w') as f:
        f.write("\n".join(commands) + "\n")

def run_multiwfn(molden_path: str, output_dir: str):
    """
    Runs Multiwfn with the generated script.
    Assumes 'Multiwfn' is in PATH.
    """
    script_path = os.path.join(output_dir, 'multiwfn.inp')
    create_multiwfn_script(script_path)
    
    # Multiwfn requires the input file as an argument
    # And reads commands from stdin
    # We can use subprocess to pipe the script content
    
    cmd = ['Multiwfn', molden_path]
    
    # Check if Multiwfn is in PATH
    import shutil
    if not shutil.which('Multiwfn') and not shutil.which('Multiwfn.exe'):
        raise FileNotFoundError("Multiwfn executable not found in PATH.")


    logging.info(f"Running Multiwfn on {molden_path}...")
    
    with open(script_path, 'r') as script_file:
        # We need to capture the output to see if it worked, 
        # but the plan says it produces 'output.txt' presumably as a result of the commands.
        # Actually, Multiwfn usually outputs to console, or saves files if requested.
        # The sequence 20... might save a file.
        # Typically option 2 in sub-menu might be "Save grid data to file"
        
        # We'll run it and assume the plan's sequence is correct for generating the file.
        # Multiwfn often dumps 'output.txt' if instructed.
        
        # Let's run it in the output_dir so any files are created there
        # But 'molden_path' might be elsewhere.
        
        # NOTE: Multiwfn might try to write to current directory.
        
        log_file = os.path.join(output_dir, 'multiwfn.log')
        with open(log_file, 'w') as log:
            subprocess.run(
                cmd,
                stdin=script_file,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=output_dir, # Run in results dir
                check=True
            )
            
    # Check if output.txt was created
    # The plan says "The file will be saved as output.txt"
    # We'll assume the sequence does this.
    # If not, we might need to rename whatever Multiwfn produced.
    # Standard Multiwfn grid export is often 'func1.grid' or similar, or prompt for name.
    # If the script doesn't provide a name, it might use a default.
    # Without interactive feedback, we rely on the user's plan being correct for their version of Multiwfn.
