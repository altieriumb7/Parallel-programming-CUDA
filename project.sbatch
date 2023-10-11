#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt


# Load necessary modules (if needed)
# module load your_module

# Your job execution commands
./main.out

# Additional commands (if needed)
# ./post_processing_script
