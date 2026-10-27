import { useEffect, useState } from 'react'
import Header from './components/Header'
import Footer from './components/Footer'
import BackendStatus from './components/BackendStatus'
import UploadArea from './components/UploadArea'
import ImagePreview from './components/ImagePreview'
import Button from './components/Button'
import LoadingState from './components/LoadingState'
import HowItWorks from './components/HowItWorks'
import Features from './components/Features'
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

function formatBytes(size) {
  if (!size) return ''
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / (1024 * 1024)).toFixed(1)} MB`
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [enhancedUrl, setEnhancedUrl] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')
  const [isEnhancing, setIsEnhancing] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [opencvAvailable, setOpencvAvailable] = useState(null)
  const [enhancementPipeline, setEnhancementPipeline] = useState(null)
  const [realesrganLoaded, setRealesrganLoaded] = useState(null)
  const [scaleFactor, setScaleFactor] = useState(4)
  const [originalSize, setOriginalSize] = useState(null)
  const [enhancedSize, setEnhancedSize] = useState(null)

  useEffect(() => {
    let isMounted = true

    async function fetchBackendHealth() {
      try {
        const health = await checkHealth()
        if (isMounted && health.status === 'ok') {
          setBackendStatus('connected')
          setOpencvAvailable(health.opencv_available ?? null)
          setEnhancementPipeline(health.enhancement_pipeline ?? null)
          setRealesrganLoaded(health.realesrgan_loaded ?? null)
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
    const interval = setInterval(fetchBackendHealth, 20000)

    return () => {
      isMounted = false
      clearInterval(interval)
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

  useEffect(() => {
    if (!isEnhancing) {
      setElapsedSeconds(0)
      return
    }

    const started = Date.now()
    const timer = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - started) / 1000))
    }, 250)

    return () => clearInterval(timer)
  }, [isEnhancing])

  const clearEnhanced = () => {
    setEnhancedUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl)
      return null
    })
    setEnhancedSize(null)
  }

  const handleImageSelect = (file) => {
    if (!file) return

    if (!isValidImageFile(file)) {
      setSelectedFile(null)
      clearEnhanced()
      setErrorMessage(
        `Unsupported file type. Please upload a ${ACCEPTED_FORMATS} image.`,
      )
      return
    }

    setErrorMessage('')
    clearEnhanced()
    setOriginalSize(null)
    setSelectedFile(file)
  }

  const handleReset = () => {
    setSelectedFile(null)
    clearEnhanced()
    setOriginalSize(null)
    setErrorMessage('')
  }

  const handleEnhance = async () => {
    if (!selectedFile) return

    setErrorMessage('')
    setIsEnhancing(true)
    setEnhancedSize(null)

    try {
      const blob = await enhanceImage(selectedFile, scaleFactor)
      const url = URL.createObjectURL(blob)
      setEnhancedUrl((previousUrl) => {
        if (previousUrl) URL.revokeObjectURL(previousUrl)
        return url
      })
      setBackendStatus('connected')
    } catch (error) {
      clearEnhanced()
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

  const backendReady = backendStatus === 'connected' && realesrganLoaded !== false

  return (
    <div id="top" className="flex min-h-screen flex-col bg-slate-50">
      <Header
        statusIndicator={
          <BackendStatus
            status={backendStatus}
            opencvAvailable={opencvAvailable}
            pipeline={enhancementPipeline}
            realesrganLoaded={realesrganLoaded}
          />
        }
      />

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 lg:px-8">
        <section className="mb-10 overflow-hidden rounded-3xl border border-slate-200 bg-gradient-to-br from-white via-indigo-50/40 to-slate-100 px-6 py-10 shadow-sm sm:px-10">
          <p className="text-sm font-semibold uppercase tracking-[0.16em] text-indigo-600">
            Local AI photo enhancer
          </p>
          <h1 className="mt-3 max-w-3xl text-3xl font-bold tracking-tight text-slate-900 sm:text-5xl">
            AI Image Quality Enhancer
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-slate-600 sm:text-lg">
            Upload a photo, upscale it with Real-ESRGAN, and download a sharper
            result — right in your browser.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a href="#enhance">
              <Button>Start enhancing</Button>
            </a>
            <a href="#how-it-works">
              <Button variant="secondary">See how it works</Button>
            </a>
          </div>
        </section>

        <div className="space-y-14">
          <section id="enhance" className="scroll-mt-24 space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-slate-900 sm:text-3xl">
                Enhance workspace
              </h2>
              <p className="mt-2 text-slate-600">
                Upload an image, choose a scale, convert, then download.
              </p>
            </div>

            {backendStatus === 'offline' ? (
              <div
                className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800"
                role="alert"
              >
                Backend is offline. Start it with:
                {' '}
                <code className="rounded bg-white px-1.5 py-0.5 text-xs">
                  python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
                </code>
                {' '}
                from the backend folder.
              </div>
            ) : null}

            {backendStatus === 'connected' && realesrganLoaded === false ? (
              <div
                className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
                role="alert"
              >
                Real-ESRGAN weights are missing. Run
                {' '}
                <code className="rounded bg-white px-1.5 py-0.5 text-xs">
                  python scripts/download_realesrgan.py
                </code>
                , then restart the backend.
              </div>
            ) : null}

            <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
              <UploadArea
                onImageSelect={handleImageSelect}
                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                disabled={isEnhancing}
              />
              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-sm text-slate-500">
                <p>Supported formats: {ACCEPTED_FORMATS}</p>
                {selectedFile ? (
                  <p>
                    {selectedFile.name}
                    {selectedFile.size ? ` · ${formatBytes(selectedFile.size)}` : ''}
                  </p>
                ) : null}
              </div>

              {errorMessage ? (
                <p
                  className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
                  role="alert"
                >
                  {errorMessage}
                </p>
              ) : null}
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <ImagePreview
                src={previewUrl}
                alt="Selected image preview"
                label="Original Image"
                sizeLabel={
                  originalSize ? `${originalSize.width} × ${originalSize.height}` : ''
                }
                emptyText="Upload a photo to see the original preview"
                onImageLoad={setOriginalSize}
              />

              <div className="space-y-2">
                <p className="text-sm font-semibold text-slate-700">
                  Enhanced Result
                  {enhancedSize ? (
                    <span className="ml-2 font-normal text-slate-500">
                      {enhancedSize.width} × {enhancedSize.height}
                    </span>
                  ) : null}
                </p>

                {isEnhancing ? (
                  <div className="flex h-64 items-center justify-center rounded-2xl border border-dashed border-indigo-300 bg-indigo-50 sm:h-80">
                    <LoadingState
                      message="Enhancing with Real-ESRGAN..."
                      detail={`This can take 30–90 seconds on CPU. Elapsed: ${elapsedSeconds}s`}
                    />
                  </div>
                ) : enhancedUrl ? (
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-sm">
                    <img
                      src={enhancedUrl}
                      alt="Enhanced image"
                      className="max-h-96 w-full object-contain"
                      onLoad={(event) => {
                        setEnhancedSize({
                          width: event.currentTarget.naturalWidth,
                          height: event.currentTarget.naturalHeight,
                        })
                      }}
                    />
                  </div>
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 sm:h-80">
                    <p className="px-4 text-center text-sm text-slate-500">
                      Click Convert Image to upscale and restore detail
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-4 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:flex-row sm:flex-wrap sm:items-center sm:justify-between sm:p-6">
              <label className="flex items-center gap-3 text-sm font-medium text-slate-700">
                Scale factor
                <select
                  value={scaleFactor}
                  onChange={(event) => setScaleFactor(Number(event.target.value))}
                  className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm"
                  disabled={isEnhancing}
                >
                  <option value={2}>2x · faster</option>
                  <option value={3}>3x</option>
                  <option value={4}>4x · best detail</option>
                </select>
              </label>

              <div className="flex flex-wrap gap-3">
                <Button
                  variant="ghost"
                  onClick={handleReset}
                  disabled={(!previewUrl && !enhancedUrl) || isEnhancing}
                >
                  Clear
                </Button>
                <Button
                  onClick={handleEnhance}
                  disabled={!previewUrl || isEnhancing || !backendReady}
                >
                  {isEnhancing ? 'Converting...' : 'Convert Image'}
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleDownload}
                  disabled={!enhancedUrl || isEnhancing}
                >
                  Download Result
                </Button>
              </div>
            </div>
          </section>

          <HowItWorks />
          <Features />
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default App
