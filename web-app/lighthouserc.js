module.exports = {
  ci: {
    collect: {
      startServerCommand: 'npm run start',
      url: ['http://localhost:3000'],
      numberOfRuns: 1, // keeping it fast for CI
      settings: {
        preset: 'desktop', // default is mobile, desktop is often relevant for dashboards
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};
