# Supabase Integration

This project is configured to work with Supabase.

## Configuration

The `.env` file contains two connection strings:

- `DATABASE_URL`: Uses the connection pooler (works on IPv4-only networks)
- `DATABASE_URL_DIRECT`: Direct connection (requires IPv6 connectivity)

## Connection Details

If you're having trouble connecting to Supabase:

1. **IPv6 Issues**: If your network doesn't support IPv6, use the pooler connection string (this is set as the default `DATABASE_URL`)
2. **Password Encoding**: Special characters in passwords (:, ?, @) need to be URL-encoded
3. **Region**: The pooler hostname might need to be updated based on your region

## API Usage

```python
# Import the Supabase client
from supabase_client import supabase

# Example query
def get_users():
    response = supabase.table('users').select('*').execute()
    if response.error:
        print(f"Error fetching users: {response.error}")
    return response.data
```

## Troubleshooting

- **Connection Issues**: Verify IP allowlist in Supabase Dashboard → Settings → Database
- **Auth Issues**: Check ANON_KEY in .env file
- **Prisma Errors**: Ensure schema has `@@schema` attributes and uses `multiSchema` feature
