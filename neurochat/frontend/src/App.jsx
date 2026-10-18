import BrainWidget from './components/BrainWidget.jsx'
import CalibrationModal from './components/CalibrationModal.jsx'
import ChatInterface from './components/ChatInterface.jsx'
import { useCalibration } from './hooks/useCalibration.js'
import { useEngagementStream } from './hooks/useEngagementStream.js'

export default function App() {
  const stream = useEngagementStream()
  const calibration = useCalibration(stream)
  const calibrated = calibration.state === 'complete'

  return (
    <>
      <BrainWidget connected={stream.connected} confidence={stream.confidence} fused={stream.fused} />
      <ChatInterface
        score={stream.fused}
        connected={stream.connected}
        calibrated={calibrated}
        onInputFocus={stream.freeze}
        onMessageSent={stream.unfreeze}
      />
      {stream.connected && !calibrated && (
        <CalibrationModal state={calibration.state} secondsLeft={calibration.secondsLeft} onStart={calibration.start} />
      )}
    </>
  )
}
