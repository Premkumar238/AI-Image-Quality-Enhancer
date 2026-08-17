const API_BASE_URL = import.meta.env.VITE_API_BASE_URL

export function getApiBaseUrl() {
  return API_BASE_URL
}

export async function checkHealth() {
  if (!API_BASE_URL) {
    throw new Error('VITE_API_BASE_URL is not configured')
  }

  const response = await fetch(`${API_BASE_URL}/api/health`)

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`)
  }

  return response.json()
}
