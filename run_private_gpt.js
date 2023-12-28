const { exec } = require('child_process');

const command = 'PGPT_PROFILES=local CUDA_VISIBLE_DEVICES=0 poetry run python -m private_gpt';

const child = exec(command);

child.stdout.on('data', (data) => {
  console.log(`stdout: ${data}`);
});

child.stderr.on('data', (data) => {
  console.error(`stderr: ${data}`);
});

child.on('close', (code) => {
  console.log(`child process exited with code ${code}`);
});
