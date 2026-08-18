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

export async function enhanceImage(file, scaleFactor = 4) {
  if (!API_BASE_URL) {
    throw new Error('VITE_API_BASE_URL is not configured')
  }

  const formData = new FormData()
  formData.append('file', file)
  formData.append('scale_factor', String(scaleFactor))

  let response
  try {
    response = await fetch(`${API_BASE_URL}/api/enhance`, {
      method: 'POST',
      body: formData,
    })
  } catch {
    throw new Error(
      'Unable to reach the backend. Start it with: python -m uvicorn app.main:app --reload (from the backend folder).',
    )
  }

  if (!response.ok) {
    let detail = `Enhancement failed with status ${response.status}`
    try {
      const errorBody = await response.json()
      if (errorBody.detail) {
        detail = errorBody.detail
      }
    } catch {
      // Keep the default error message when the body is not JSON.
    }
    throw new Error(detail)
  }

  const blob = await response.blob()
  return blob
}
