import { useEffect, useState } from 'react'
import Header from './components/Header'
import BackendStatus from './components/BackendStatus'
import UploadArea from './components/UploadArea'
import ImagePreview from './components/ImagePreview'
import Button from './components/Button'
import { checkHealth } from './services/api'

const ALLOWED_TYPES = ['image/jpeg', 'image/png']
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']
const ACCEPTED_FORMATS = 'JPG, JPEG, and PNG'

function isValidImageFile(file) {
  const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  return ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTENSIONS.includes(extension)
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [showResult, setShowResult] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    let isMounted = true

    async function fetchBackendHealth() {
      try {
        const health = await checkHealth()
        if (isMounted && health.status === 'ok') {
          setBackendStatus('connected')
        } else if (isMounted) {
          setBackendStatus('offline')
        }
      } catch {
        if (isMounted) {
          setBackendStatus('offline')
        }
      }
    }

    fetchBackendHealth()

    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null)
      return
    }

    const url = URL.createObjectURL(selectedFile)
    setPreviewUrl(url)

    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  const handleImageSelect = (file) => {
    if (!file) return

    if (!isValidImageFile(file)) {
      setSelectedFile(null)
      setShowResult(false)
      setErrorMessage(
        `Unsupported file type. Please upload a ${ACCEPTED_FORMATS} image.`,
      )
      return
    }

    setErrorMessage('')
    setShowResult(false)
    setSelectedFile(file)
  }

  const handleEnhance = () => {
    if (!previewUrl) return
    setShowResult(true)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header statusIndicator={<BackendStatus status={backendStatus} />} />

      <main className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
        <section className="mb-8 text-center">
          <h2 className="text-3xl font-bold text-gray-900 sm:text-4xl">
            AI Image Enhancer
          </h2>
          <p className="mt-3 text-base text-gray-600 sm:text-lg">
            Enhance low-quality images using deep learning.
          </p>
        </section>

        <div className="space-y-6">
          <section>
            <UploadArea
              onImageSelect={handleImageSelect}
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            />
            <p className="mt-2 text-center text-sm text-gray-500">
              Supported formats: {ACCEPTED_FORMATS}
            </p>
            {errorMessage && (
              <p
                className="mt-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
                role="alert"
              >
                {errorMessage}
              </p>
            )}
          </section>

          <section className="grid gap-6 md:grid-cols-2">
            <ImagePreview
              src={previewUrl}
              alt="Selected image preview"
              label="Selected Image"
            />

            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700">Enhanced Result</p>
              {showResult ? (
                <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-indigo-300 bg-indigo-50 px-4 text-center sm:h-80">
                  <p className="text-sm text-indigo-700">
                    Enhanced image will appear here once the backend and ML
                    model are connected.
                  </p>
                </div>
              ) : (
                <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-gray-300 bg-gray-50 sm:h-80">
                  <p className="text-sm text-gray-500">
                    Click &quot;Enhance Image&quot; to see the result section
                  </p>
                </div>
              )}
            </div>
          </section>

          <section className="flex justify-center">
            <Button onClick={handleEnhance} disabled={!previewUrl}>
              Enhance Image
            </Button>
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
