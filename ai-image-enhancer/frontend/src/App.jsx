import { useEffect, useState } from 'react'
import Header from './components/Header'
import BackendStatus from './components/BackendStatus'
import UploadArea from './components/UploadArea'
import ImagePreview from './components/ImagePreview'
import Button from './components/Button'
import LoadingState from './components/LoadingState'
import { checkHealth, enhanceImage } from './services/api'

const ALLOWED_TYPES = ['image/jpeg', 'image/png']
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']
const ACCEPTED_FORMATS = 'JPG, JPEG, and PNG'

function isValidImageFile(file) {
  const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  return ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTENSIONS.includes(extension)
}

function getEnhancedFilename(originalName) {
  const dotIndex = originalName.lastIndexOf('.')
  if (dotIndex === -1) {
    return `enhanced_${originalName}.png`
  }
  return `enhanced_${originalName.slice(0, dotIndex)}${originalName.slice(dotIndex)}`
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [enhancedUrl, setEnhancedUrl] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [scaleFactor, setScaleFactor] = useState(4)

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

  useEffect(() => {
    return () => {
      if (enhancedUrl) {
        URL.revokeObjectURL(enhancedUrl)
      }
    }
  }, [enhancedUrl])

  const handleImageSelect = (file) => {
    if (!file) return

    if (!isValidImageFile(file)) {
      setSelectedFile(null)
      setEnhancedUrl(null)
      setErrorMessage(
        `Unsupported file type. Please upload a ${ACCEPTED_FORMATS} image.`,
      )
      return
    }

    setErrorMessage('')
    setEnhancedUrl(null)
    setSelectedFile(file)
  }

  const handleEnhance = async () => {
    if (!selectedFile) return

    setErrorMessage('')
    setIsEnhancing(true)

    try {
      const blob = await enhanceImage(selectedFile, scaleFactor)
      const url = URL.createObjectURL(blob)
      setEnhancedUrl((previousUrl) => {
        if (previousUrl) {
          URL.revokeObjectURL(previousUrl)
        }
        return url
      })
    } catch (error) {
      setEnhancedUrl(null)
      setBackendStatus('offline')
      setErrorMessage(
        error instanceof Error
          ? error.message
          : 'Unable to enhance the image. Please try again.',
      )
    } finally {
      setIsEnhancing(false)
    }
  }

  const handleDownload = () => {
    if (!enhancedUrl || !selectedFile) return

    const link = document.createElement('a')
    link.href = enhancedUrl
    link.download = getEnhancedFilename(selectedFile.name)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
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
            Upload an image, convert it with the trained SRCNN model, and download the result.
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
              label="Original Image"
            />

            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-700">Converted Result</p>
              {isEnhancing ? (
                <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-indigo-300 bg-indigo-50 sm:h-80">
                  <LoadingState message="Converting your image..." />
                </div>
              ) : enhancedUrl ? (
                <div className="overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
                  <img
                    src={enhancedUrl}
                    alt="Converted image"
                    className="max-h-96 w-full object-contain"
                  />
                </div>
              ) : (
                <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-gray-300 bg-gray-50 sm:h-80">
                  <p className="text-sm text-gray-500">
                    Click &quot;Convert Image&quot; to enhance the uploaded photo
                  </p>
                </div>
              )}
            </div>
          </section>

          <section className="flex flex-col items-center justify-center gap-4 sm:flex-row">
            <label className="flex items-center gap-2 text-sm text-gray-700">
              Scale factor
              <select
                value={scaleFactor}
                onChange={(event) => setScaleFactor(Number(event.target.value))}
                className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm"
                disabled={isEnhancing}
              >
                <option value={2}>2x</option>
                <option value={3}>3x</option>
                <option value={4}>4x</option>
              </select>
            </label>
            <Button onClick={handleEnhance} disabled={!previewUrl || isEnhancing}>
              {isEnhancing ? 'Converting...' : 'Convert Image'}
            </Button>
            <Button onClick={handleDownload} disabled={!enhancedUrl || isEnhancing}>
              Download Result
            </Button>
          </section>
        </div>
      </main>
    </div>
  )
}

export default App
