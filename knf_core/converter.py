import os
import logging
from .utils import run_subprocess

def ensure_xyz(input_path: str, output_dir: str) -> str:
    """
    Ensures the input file is in XYZ format.
    If it's already .xyz, returns the path (copied to output_dir).
    If not, converts it using obabel.
    """
    filename = os.path.basename(input_path)
    base, ext = os.path.splitext(filename)
    ext = ext.lower()
    
    target_xyz = os.path.join(output_dir, f"{base}.xyz")
    
    if ext == '.xyz':
        # Just copy if it's not already there
        if not os.path.exists(target_xyz):
            # We assume the caller handles copying if needed, 
            # but here we can ensure it exists in output_dir
            # For simplicity, if input is already .xyz and matches target, done.
            if os.path.abspath(input_path) != os.path.abspath(target_xyz):
                import shutil
                shutil.copy2(input_path, target_xyz)
        return target_xyz
        
    # Conversion needed
    logging.info(f"Converting {filename} to XYZ using Open Babel...")
    
    # obabel -i<format> <input> -oxyz -O <output>
    # format is derived from extension (without dot)
    informat = ext[1:]
    
    cmd = ['obabel',f'-i{informat}', input_path, '-oxyz', '-O', target_xyz]
    
    run_subprocess(cmd)
    
    if not os.path.exists(target_xyz):
        raise RuntimeError(f"Open Babel failed to convert {input_path} to {target_xyz}")
        
    return target_xyz
