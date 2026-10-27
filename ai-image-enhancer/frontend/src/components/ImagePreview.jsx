function ImagePreview({
  src,
  alt = 'Preview',
  label = 'Preview',
  sizeLabel,
  emptyText = 'No image to preview',
  onImageLoad,
}) {
  return (
    <div className="space-y-2">
      {label ? (
        <p className="text-sm font-semibold text-slate-700">
          {label}
          {sizeLabel ? (
            <span className="ml-2 font-normal text-slate-500">{sizeLabel}</span>
          ) : null}
        </p>
      ) : null}

      {src ? (
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-sm">
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
      ) : (
        <div className="flex h-64 items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 sm:h-80">
          <p className="px-4 text-center text-sm text-slate-500">{emptyText}</p>
        </div>
      )}
    </div>
  )
}

export default ImagePreview
