import { useState, useRef } from "react";

export default function Dropzone({
  onFileSelect,
  accept = "image/*",
  label = "Drop an image here or click to browse",
  file,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFileSelect(dropped);
  };

  const handleChange = (e) => {
    const selected = e.target.files[0];
    if (selected) onFileSelect(selected);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
        isDragging
          ? "border-accent bg-accent/5"
          : "border-gray-300 hover:border-accent hover:bg-gray-50"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
      />
      <div className="flex flex-col items-center gap-3">
        <div className="w-14 h-14 rounded-full bg-black flex items-center justify-center">
          <svg
            className="w-7 h-7 text-accent"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
        </div>
        {file ? (
          <div>
            <p className="font-medium text-black">{file.name}</p>
            <p className="text-sm text-gray-500 mt-1">
              {(file.size / 1024 / 1024).toFixed(2)} MB &middot; Click or drop to replace
            </p>
          </div>
        ) : (
          <>
            <p className="font-medium text-black">{label}</p>
            <p className="text-sm text-gray-500">
              Drag and drop, or click to select a file
            </p>
          </>
        )}
      </div>
    </div>
  );
}
