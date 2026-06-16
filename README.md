# iFragDI

iFragDI is a standalone Python pipeline for partner-specific interface-residue scoring and docking-restraint generation.

This repository tracks the working source code, lightweight benchmark orchestration scripts, focused local tests, and documentation needed to review and develop the pipeline. Raw benchmark data, generated benchmark outputs, databases, matrices, images, and local execution logs are intentionally excluded from version control.

Heavy benchmark execution is manual on Shiva through Slurm. 

Current repository state: working snapshot under review. It is not a final validated release and should not be described as scientifically stable. This snapshot includes the current unverified `--allow-no-evidence` conservation handling patch in `conservation.py` and `combine_ifrag_radi.py`.
