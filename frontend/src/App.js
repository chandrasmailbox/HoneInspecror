import React, { useState, useRef } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const VideoUpload = ({ onAnalysisComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
      } else {
        alert('Please select a video file');
      }
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const analyzeVideo = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/analyze-video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onAnalysisComplete(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">HomeInspector AI</h1>
        <p className="text-gray-600">Upload a video to detect house defects using AI</p>
      </div>

      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="space-y-4">
          <div className="text-6xl text-gray-400">📹</div>
          <div>
            <p className="text-lg font-medium text-gray-700">
              {selectedFile ? selectedFile.name : 'Drop your video here or click to select'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Supports MP4, AVI, MOV, MKV files
            </p>
          </div>
        </div>
      </div>

      {selectedFile && (
        <div className="mt-6 text-center">
          <button
            onClick={analyzeVideo}
            disabled={isAnalyzing}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-3 px-8 rounded-lg transition-colors"
          >
            {isAnalyzing ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing Video...
              </span>
            ) : (
              'Analyze Video'
            )}
          </button>
        </div>
      )}
    </div>
  );
};

const ResultsDisplay = ({ results, onBack }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getDefectIcon = (type) => {
    switch (type) {
      case 'cracks': return '🔧';
      case 'water_damage': return '💧';
      case 'mold on wall': return '🦠';
      case 'paint peeling off wall': return '🎨';
      default: return '⚠️';
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold text-gray-800">Inspection Results</h2>
        <button
          onClick={onBack}
          className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg"
        >
          ← Back to Upload
        </button>
      </div>

      {/* Summary Card */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Inspection Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{results.summary.total_defects_found}</div>
            <div className="text-sm text-gray-600">Total Defects</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{results.summary.frames_analyzed}</div>
            <div className="text-sm text-gray-600">Frames Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{results.summary.high_confidence_detections}</div>
            <div className="text-sm text-gray-600">High Confidence</div>
          </div>
          <div className="text-center">
            <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(results.summary.severity)}`}>
              {results.summary.severity.toUpperCase()}
            </span>
            <div className="text-sm text-gray-600 mt-1">Severity</div>
          </div>
        </div>
      </div>

      {/* Defect Types */}
      {results.summary.defect_types.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Detected Issues</h3>
          <div className="flex flex-wrap gap-2">
            {results.summary.defect_types.map((type, index) => (
              <span
                key={index}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800"
              >
                {getDefectIcon(type)} {type.replace('_', ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Frame Analysis */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Frame-by-Frame Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {results.defects_found
            .filter(frame => frame.defects.length > 0)
            .slice(0, 12)
            .map((frame, index) => (
            <div
              key={index}
              className="border rounded-lg p-4 cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => setSelectedFrame(frame)}
            >
              <img
                src={`data:image/jpeg;base64,${frame.frame_image}`}
                alt={`Frame ${frame.frame_number}`}
                className="w-full h-32 object-cover rounded mb-2"
              />
              <div className="text-sm text-gray-600">Frame {frame.frame_number}</div>
              <div className="text-sm font-medium">
                Confidence: {(frame.confidence_score * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {frame.defects.length} defect(s) found
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Selected Frame Modal */}
      {selectedFrame && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl max-h-screen overflow-y-auto p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold">Frame {selectedFrame.frame_number} Details</h3>
              <button
                onClick={() => setSelectedFrame(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl"
              >
                ×
              </button>
            </div>
            
            <img
              src={`data:image/jpeg;base64,${selectedFrame.frame_image}`}
              alt={`Frame ${selectedFrame.frame_number}`}
              className="w-full max-w-md mx-auto rounded mb-4"
            />
            
            <div className="space-y-3">
              <h4 className="font-semibold">Detected Defects:</h4>
              {selectedFrame.defects.map((defect, index) => (
                <div key={index} className="bg-gray-50 p-3 rounded">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{getDefectIcon(defect.type)} {defect.type}</span>
                    <span className="text-sm text-gray-600">
                      {(defect.confidence * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  {defect.description && (
                    <p className="text-sm text-gray-600 mt-1">{defect.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
  };

  const handleBackToUpload = () => {
    setAnalysisResults(null);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {!analysisResults ? (
        <VideoUpload onAnalysisComplete={handleAnalysisComplete} />
      ) : (
        <ResultsDisplay results={analysisResults} onBack={handleBackToUpload} />
      )}
    </div>
  );
}

export default App;
