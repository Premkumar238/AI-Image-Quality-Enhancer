import { useState } from 'react'

function UploadArea({ onImageSelect, accept = 'image/*' }) {
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = (file) => {
    if (file) {
      onImageSelect?.(file)
    }
  }

  const handleInputChange = (event) => {
    const file = event.target.files?.[0]
    handleFile(file)
  }

  const handleDragOver = (event) => {
    event.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files?.[0]
    handleFile(file)
  }

  return (
    <label
      htmlFor="image-upload"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-12 transition ${
        isDragging
          ? 'border-indigo-500 bg-indigo-50'
          : 'border-gray-300 bg-gray-50 hover:border-indigo-400 hover:bg-indigo-50'
      }`}
    >
      <input
        id="image-upload"
        type="file"
        accept={accept}
        onChange={handleInputChange}
        className="hidden"
      />
      <p className="text-base font-medium text-gray-700">
        Drag and drop an image here
      </p>
      <p className="mt-1 text-sm text-gray-500">or click to select a file</p>
    </label>
  )
}

export default UploadArea
