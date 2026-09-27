import { createClient } from '@supabase/supabase-js'

const url = process.env.NEXT_PUBLIC_SUPABASE_URL
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

if (!url || !anonKey) {
  // Fail loudly in dev; on Vercel these env vars are required for the build.
  console.warn(
    '[supabase] NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY not set'
  )
}

/**
 * Browser client (anon key, read-only via RLS policies).
 * Writes happen only through /api/* routes using the service key server-side.
 */
export const supabase = createClient(
  url || 'http://localhost:54321',
  anonKey || 'public-anon-key',
  {
    auth: { persistSession: false },
    realtime: { params: { eventsPerSecond: 5 } },
  }
)

/**
 * Server-side client for API routes (Node runtime) — uses the service key.
 * NEVER import this from client components; service key must stay server-only.
 */
export function supabaseAdmin() {
  const serviceKey = process.env.SUPABASE_SERVICE_KEY
  if (!serviceKey) {
    throw new Error('SUPABASE_SERVICE_KEY is not set (server-side only)')
  }
  return createClient(url, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  })
}
