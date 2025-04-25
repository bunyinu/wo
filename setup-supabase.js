// Setup Supabase for Moyo project
const { setupSupabase } = require('/home/lus/supabase-windsurf-template.js');

// Configuration for Moyo project
const config = {
  projectDir: '/home/lus/Downloads/moyo',
  projectRef: 'dpdifpribtulgpsyfxiw',  // Actual Supabase project reference
  dbPassword: 'postgres',              // Default database password
  anonKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRwZGlmcHJpYnR1bGdwc3lmeGl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU1NjY3NDUsImV4cCI6MjA2MTE0Mjc0NX0.e9khagck5XGiAlgVi_intB60pn7K5qL8PKzPcH-OGMQ',  // Actual anon key
  usePrisma: false,                // Moyo uses Supabase SDK directly, not Prisma
  useTypescript: true,             // Frontend is in TypeScript
};

// Run the setup
setupSupabase(config);
