!/bin/bash
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "To connect:
ssh -N -L ${port}:${node}:${port} ${user}@hypatia-login.hpc.phys.ucl.ac.uk

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)

Remember to scancel job when done. Check output below for access token if
you need it.
"
conda activate instaseis
jupyter-notebook --no-browser --port=${port} --ip=${node}