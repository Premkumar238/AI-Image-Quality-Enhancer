function ImagePreview({ src, alt = 'Preview', label = 'Preview', sizeLabel, onImageLoad }) {
  if (!src) {
    return (
      <div className="space-y-2">
        {label && <p className="text-sm font-medium text-gray-700">{label}</p>}
        <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-gray-300 bg-gray-50">
          <p className="text-sm text-gray-500">No image to preview</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {label && (
        <p className="text-sm font-medium text-gray-700">
          {label}
          {sizeLabel ? <span className="ml-2 font-normal text-gray-500">{sizeLabel}</span> : null}
        </p>
      )}
      <div className="overflow-hidden rounded-lg border border-gray-200 bg-gray-50">
        <img
          src={src}
          alt={alt}
          className="max-h-96 w-full object-contain"
          onLoad={(event) => {
            onImageLoad?.({
              width: event.currentTarget.naturalWidth,
              height: event.currentTarget.naturalHeight,
            })
          }}
        />
      </div>
    </div>
  )
}

export default ImagePreview
