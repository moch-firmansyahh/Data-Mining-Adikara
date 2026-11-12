import nbformat as nbf
import io
import sys
import os
import contextlib

def execute_notebook_in_process(nb_path):
    print(f"Reading {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    global_scope = {}
    
    # We change directory to 'notebooks' so relative paths like '../data' or 'data' resolve properly
    orig_cwd = os.getcwd()
    os.chdir('notebooks')
    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                print(f"Executing code cell {idx+1}/{len(nb.cells)}...")
                source = cell.source
                stdout_buf = io.StringIO()
                stderr_buf = io.StringIO()
                try:
                    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                        exec(source, global_scope)
                    
                    stdout_str = stdout_buf.getvalue()
                    stderr_str = stderr_buf.getvalue()
                    cell.outputs = []
                    if stdout_str:
                        cell.outputs.append(nbf.v4.new_output(output_type='stream', name='stdout', text=stdout_str))
                    if stderr_str:
                        cell.outputs.append(nbf.v4.new_output(output_type='stream', name='stderr', text=stderr_str))
                    cell.execution_count = idx + 1
                except Exception as e:
                    print(f"ERROR executing cell {idx+1}: {e}")
                    raise
                    
        print("All code cells executed successfully!")
    finally:
        os.chdir(orig_cwd)
        
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully saved executed notebook with all cell outputs to {nb_path}!")

if __name__ == '__main__':
    execute_notebook_in_process('notebooks/final_submission_pipeline.ipynb')

