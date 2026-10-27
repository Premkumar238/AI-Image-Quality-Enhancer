import { useState } from 'react'

function UploadArea({ onImageSelect, accept = 'image/*', disabled = false }) {
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = (file) => {
    if (!disabled && file) {
      onImageSelect?.(file)
    }
  }

  const handleInputChange = (event) => {
    const file = event.target.files?.[0]
    handleFile(file)
    event.target.value = ''
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    if (!disabled) setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    if (disabled) return
    const file = event.dataTransfer.files?.[0]
    handleFile(file)
  }

  return (
    <label
      htmlFor="image-upload"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-14 transition ${
        disabled ? 'cursor-not-allowed opacity-60' : ''
      } ${
        isDragging
          ? 'border-indigo-500 bg-indigo-50'
          : 'border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-indigo-50/60'
      }`}
    >
      <input
        id="image-upload"
        type="file"
        accept={accept}
        onChange={handleInputChange}
        disabled={disabled}
        className="hidden"
      />
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-100 text-indigo-600">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          className="h-7 w-7"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5V19a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-2.5M16 8l-4-4m0 0L8 8m4-4v12"
          />
        </svg>
      </div>
      <p className="text-base font-semibold text-slate-800">
        Drag and drop an image here
      </p>
      <p className="mt-1 text-sm text-slate-500">or click to browse JPG / PNG files</p>
    </label>
  )
}

export default UploadArea
