module.exports = {
  apps: [
    {
      name: 'private-gpt', // choose a name for your app
      script: 'poetry',
      args: ['run', 'poetry', 'run', 'python', '-m', 'private_gpt'],
      interpreter_args: 'run', // additional arguments for the interpreter
      autorestart: true,
      watch: true, // optional: enable file watching for automatic restart on file changes
      env: {
        NODE_ENV: 'production', // set the environment variables if needed
        PGPT_PROFILES: 'local',
        CUDA_VISIBLE_DEVICES: '0',
      },
    },
  ],
};
