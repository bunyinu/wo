import { createBrowserClient } from "@supabase/ssr";

// Using hardcoded mock values to bypass environment variables requirement
export const createClient = () =>
  createBrowserClient(
    "https://example.supabase.co",  // Mock Supabase URL
    "mock-anon-key",               // Mock Anon Key
  );
