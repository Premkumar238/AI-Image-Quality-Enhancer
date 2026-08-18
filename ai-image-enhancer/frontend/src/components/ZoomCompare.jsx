import { useEffect, useState } from 'react'

function cropCenter(image, size = 360) {
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext('2d')
  if (!context) {
    return null
  }

  const sourceSize = Math.min(image.naturalWidth, image.naturalHeight, size)
  const sx = Math.max(0, Math.floor((image.naturalWidth - sourceSize) / 2))
  const sy = Math.max(0, Math.floor((image.naturalHeight - sourceSize) / 2))
  context.imageSmoothingEnabled = false
  context.drawImage(image, sx, sy, sourceSize, sourceSize, 0, 0, size, size)
  return canvas.toDataURL('image/png')
}

function loadCrop(src) {
  return new Promise((resolve) => {
    const image = new Image()
    image.onload = () => resolve(cropCenter(image))
    image.src = src
  })
}

function ZoomCompare({ originalSrc, enhancedSrc }) {
  const [originalCrop, setOriginalCrop] = useState(null)
  const [enhancedCrop, setEnhancedCrop] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function createCrops() {
      const [original, enhanced] = await Promise.all([
        loadCrop(originalSrc),
        loadCrop(enhancedSrc),
      ])
      if (!cancelled) {
        setOriginalCrop(original)
        setEnhancedCrop(enhanced)
      }
    }

    createCrops()
    return () => {
      cancelled = true
    }
  }, [originalSrc, enhancedSrc])

  if (!originalSrc || !enhancedSrc) {
    return null
  }

  return (
    <section className="rounded-lg border border-gray-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-gray-800">100% zoom (center crop)</h3>
      <p className="mt-1 text-xs text-gray-500">
        The two previews above are fitted to the same box, so they can look similar.
        This crop shows actual pixels so sharpness and detail are easier to compare.
      </p>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <p className="mb-2 text-xs font-medium text-gray-600">Original zoom</p>
          {originalCrop ? (
            <img
              src={originalCrop}
              alt="Original zoom crop"
              className="h-72 w-full rounded-lg border border-gray-200 object-cover"
            />
          ) : (
            <div className="h-72 rounded-lg border border-dashed border-gray-300 bg-gray-50" />
          )}
        </div>
        <div>
          <p className="mb-2 text-xs font-medium text-gray-600">Enhanced zoom</p>
          {enhancedCrop ? (
            <img
              src={enhancedCrop}
              alt="Enhanced zoom crop"
              className="h-72 w-full rounded-lg border border-gray-200 object-cover"
            />
          ) : (
            <div className="h-72 rounded-lg border border-dashed border-gray-300 bg-gray-50" />
          )}
        </div>
      </div>
    </section>
  )
}

export default ZoomCompare
