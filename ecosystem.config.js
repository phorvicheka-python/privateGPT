module.exports = {
  apps: [
    {
      name: 'private-gpt',
      script: './run_private_gpt.sh',
      autorestart: true,
      watch: true,
      env: {
        PGPT_PROFILES: 'local',
        CUDA_VISIBLE_DEVICES: '0',
      },
    },
  ],
};
