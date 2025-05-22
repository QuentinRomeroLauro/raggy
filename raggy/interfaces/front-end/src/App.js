import './App.css';
import { useEffect, useState } from 'react';
import { ToastContainer, toast, Bounce } from 'react-toastify';
import PillButton from './components/PillButton';
import LLMBox from './components/LLMBox';
import RetrievalBox from './components/RetrievalBox';
import QueryBox from './components/QueryBox';
import AnswerBox from './components/AnswerBox';
import LoadingBox from './components/LoadingBox';
import io from 'socket.io-client';
import 'react-toastify/dist/ReactToastify.css';
import ListParameter from './components/ListParameter';
  
  let retries = 5;
  const socket = io('http://localhost:5001', {
      reconnectionAttempts: retries,
      timeout: 10000,
      transports: ['websocket', 'polling']
    });


function App() {
  const [steps, setSteps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState();
  const [answer, setAnswer] = useState();
  const [traces, setTraces] = useState([]);
  const [selectedTrace, setSelectedTrace] = useState();
  const [semanticSimilarity, setSemanticSimilarity] = useState();

  const [openCards, setOpenCards] = useState([0, steps.length - 1]);

  useEffect(() => {
    const handleLLM = (data) => {
      console.log('Received LLM call:', data);
      const step = {
        type: 'LLM',
        promptText: data.promptText,
        responseText: data.responseText,
        temperature: data.temperature,
        maxTokens: data.maxTokens,
        id: data.id,
        order: data.order,
      };
      setSteps((prevSteps) => [...prevSteps, step]);
      setBoxColors((prevBoxColors) => [...prevBoxColors, 'f0f0f0']);
    };

    const handleRetrieval = (data) => {
      console.log('Received Retrieval call:', data);

      console.log("recieved selectedChunks: ", data.selectedChunks);

      const step = {
        type: 'Retrieval',
        query: data.query,
        vanillaChunks: data.vanillaChunks,
        raptorChunks: data.raptorChunks,
        selectedChunks: data.selectedChunks,
        retrievalMode: data.retrievalMode,
        searchBy: data.searchBy,
        chunkSize: data.chunkSize,
        chunkOverlap: data.chunkOverlap,
        k: data.k,
        id: data.id,
        order: data.order,
      };
      setSteps((prevSteps) => [...prevSteps, step]);
      setBoxColors((prevBoxColors) => [...prevBoxColors, 'f0f0f0']);
    };

    const handleQuery = (data) => {

      const newSteps = steps.filter((step) => step.order < data.order);
      setSteps(newSteps);
      // set all the colors to default
      setBoxColors((prevBoxColors) => prevBoxColors.map(() => 'f0f0f0'));

      console.log('Received Query:', data);
      const step = {
        type: 'Query',
        query: data.query,
        id: data.id,
        order: data.order,
      };
      setQuery(data.query);

      setSteps((prevSteps) => {
        if (prevSteps.length > 0) {
          console.log("Steps before clearing:", prevSteps);
          console.log("clearing steps");
          return [step];
        } else {
          console.log("Adding step to empty steps:", step);
          return [...prevSteps, step];
        }
      });
      setBoxColors((prevBoxColors) => [...prevBoxColors, 'f0f0f0']);
    };

    const handleAnswer = (data) => {
      console.log('Received answer:', data.answer);
      const step = {
        type: 'Answer',
        answer: data.answer,
        id: data.id,
        order: data.order,
      };
      setAnswer(data.answer);
      setSteps((prevSteps) => [...prevSteps, step]);
      setOpenCards((prevOpenCards) => [...prevOpenCards, steps.length]);
      setBoxColors((prevBoxColors) => [...prevBoxColors, 'f0f0f0']);
    };

    const handleTraces = (data) => {
      console.log('Received trace:', data);
      // get the saved traces from the server
      fetch('http://localhost:5001/get_traces')
        .then((response) => response.json())
        .then((data) => {
          setTraces((traces) => {
            const parsedData = data.map(element => JSON.parse(element));  
            console.log('Received traces:', parsedData);  
            if (selectedTrace === undefined && parsedData.length > 0) {
              console.log("setting selected trace to first trace");
              setSelectedTrace(parsedData[0]);
            }
            return parsedData;
          });
        })
        .catch((error) => {
          console.error('Error fetching traces:', error);
        });
    };


    const handleLoading = (data) => {
      console.log('Received loading:', data.loading);
      setLoading(data.loading);
    };

    socket.on('connect', () => {
      console.log('Connected to server\nWe can run connection logic here; e.g. running the pipeline for the first time to get an outline');

      fetch('http://localhost:5001/get_traces')
        .then((response) => response.json())
        .then((data) => {
          setTraces((traces) => {
            const parsedData = data.map(element => JSON.parse(element));  
            console.log('Received traces:', parsedData);  
            if (selectedTrace === undefined) {
              console.log("setting selected trace to first trace");
              setSelectedTrace(parsedData[0]);
            }
            return parsedData
          });
        })
        .catch((error) => {
          console.error('Error fetching traces:', error);
        });
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from server');
    });


    socket.on('llm_data', handleLLM);
    socket.on('traces', handleTraces);
    socket.on('retrieval_data', handleRetrieval);
    socket.on('query_data', handleQuery);
    socket.on('answer_data', handleAnswer);
    socket.on('loading', handleLoading);

    return () => {
      socket.off('connect');
      socket.off('llm_data', handleLLM);
      socket.off('retrieval_data', handleRetrieval);
      socket.off('query_data', handleQuery);
      socket.off('answer_data', handleAnswer);
      socket.off('loading', handleLoading);
      socket.off('disconnect');
    };
  }, [steps]);


  const handleSetCollapsed = (index) => {
    setOpenCards((prevOpenCards) => {
      if (prevOpenCards.includes(index)) {
        return prevOpenCards.filter((i) => i !== index);
      } else {
        return [...prevOpenCards, index];
      }
    });
  }

  const isCollapsed = (index) => {
    return !openCards.includes(index);
  }

  const [boxColors, setBoxColors] = useState([]);
  const handleColorChanges = (index) => {
    // change all the colors at index greater than the current index to '#fcb3b3'
    const newColors = boxColors.map((color, i) => {
      if (i > index) {
        return '#fcb3b3';
      } else {
        return color;
      }
    }
    );
    setBoxColors(newColors);
  }

  const handleSetBoxColor = (index, color) => {
    const newColors = boxColors.map((c, i) => {
      if (i === index) {
        return color;
      } else {
        return c;
      }
    });
    setBoxColors(newColors);
  }


  const handleFinishRunningPipeline = async (data, i) => {
    setLoading(true);
    console.log('handleFinishRunningPipeline called with step:', data);

    // remove all steps that were ahead of the step that this was called on
    const newSteps = steps.filter((step) => step.order <= data.order);
    setSteps(newSteps);

    // send the data to the server
    const response = await fetch('http://localhost:5001/finish_running_pipeline', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (response.ok) {
      console.log('Finished running pipeline successfully');
      const responseData = await response.json();
      console.log('Data:', responseData);
      // handle the responseData if necessary
    }
    console.log('i', i)
    handleColorChanges(i);
    if (data.step === 'Query') {
      setBoxColors((prevBoxColors) => prevBoxColors.map(() => 'f0f0f0'));
    }


    setLoading(false);
    setSemanticSimilarity();
  };

  console.log("steps", steps);

  const handleSaveTrace = () => {
    console.log('saving global trace');
    const data = {
      type: 'Answer',
      query: query,
      answer: answer,
    };

    // send the data to the server
    fetch('http://localhost:5001/save_trace', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    toast('answer saved ✅', {
      position: "top-left",
      autoClose: 3000,
      hideProgressBar: true,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      progress: undefined,
    });

  };

  const handleCheckSimilarity = async () => {
    if (!selectedTrace || !answer) {
      toast.error("Need both a trace and an answer to check similarity", {
        position: "top-left",
        autoClose: 3000,
        hideProgressBar: true,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
      });
      return;
    }

    toast('checking similarity...', {
      position: "top-left",
      autoClose: 3000,
      hideProgressBar: true,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      progress: undefined,
    });

    const query = selectedTrace.input || selectedTrace.query;
    
    const response = await fetch('http://localhost:5001/get_evaluation_embedding_distance', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        query: query,
        answer: answer,
      }),
    });

    if (response.ok) {
      const responseData = await response.json();
      console.log('Semantic Similarity:', responseData);
      setSemanticSimilarity(responseData);
    }
  };

  return (
    <>
      <div className="App">

        <header className="App-header">
          <div className="App-header-left">
            <h3>raggy 🐶</h3>
          </div>

          {/* Traces */}
          <div className="App-header-right" style={{ 
            marginRight: '20px', 
            marginLeft: '20px',
            display: 'flex', 
            alignItems: 'center',
            flex: 1,
            justifyContent: 'flex-end'
          }}>
            {traces.length > 0 &&
              <>
                <select 
                  value={selectedTrace?.id} 
                  onChange={(e) => {
                    const selectedValue = e.target.value;
                    console.log('selected trace:', selectedValue);
                    const selectedTraceObj = traces.find(trace => trace.id.toString() === selectedValue);
                    if(selectedTraceObj !== undefined) {
                      setSelectedTrace(selectedTraceObj);
                    }
                  }}
                  style={{
                    padding: '8px 12px',
                    fontSize: '14px',
                    borderRadius: '16px',
                    border: '1px solid #ccc',
                    backgroundColor: '#f8f9fa',
                    marginRight: '15px',
                    minWidth: '400px',
                    cursor: 'pointer',
                    outline: 'none',
                    height: '36px'
                  }}
                >
                  {traces.map(trace => (
                    <option 
                      key={trace.id} 
                      value={trace.id}
                      style={{
                        padding: '8px',
                        fontSize: '14px'
                      }}
                    >
                      {trace.input || trace.query || 'No input text'}
                    </option>
                  ))}
                </select>

                <PillButton
                  text="check similarity"
                  onClick={() => handleCheckSimilarity()}
                />
              </>
            }
          </div>
        </header>
        <div className='App-content'>
          <ToastContainer
            position="top-right"
            autoClose={5000}
            hideProgressBar
            newestOnTop={false}
            closeOnClick
            rtl={false}
            pauseOnFocusLoss
            draggable
            pauseOnHover
            theme="light"
            transition={Bounce}
          />


          {steps.map((step, i) => {
            console.log("i", i);
            if (step.type === 'Query') {
              return (
                <QueryBox
                  key={i}
                  number={i}
                  query={query}
                  setQuery={setQuery}
                  id={step.id}
                  order={step.order}
                  setIsCollapsed={() => handleSetCollapsed(i)}
                  isCollapsed={isCollapsed(i)}
                  handleFinishRunningPipeline={handleFinishRunningPipeline}
                />
              );
            } else if (step.type === 'Answer') {
              return (
                <AnswerBox
                  key={i}
                  answer={answer}
                  setAnswer={setAnswer}
                  id={step.id}
                  order={step.order}
                  handleSaveTrace={handleSaveTrace}
                  setIsCollapsed={() => handleSetCollapsed(i)}
                  isCollapsed={isCollapsed(i)}
                  handleFinishRunningPipeline={handleFinishRunningPipeline}
                  boxColor={boxColors[i]}
                  setBoxColor={(color) => handleSetBoxColor(i, color)}
                  semanticSimilarity={semanticSimilarity}
                />
              );
            } else if (step.type === 'LLM') {
              return (
                <LLMBox
                  key={i}
                  promptText={step.promptText}
                  responseText={step.responseText}
                  temperature={step.temperature}
                  maxTokens={step.maxTokens}
                  id={step.id}
                  order={step.order}
                  setIsCollapsed={() => handleSetCollapsed(i)}
                  isCollapsed={isCollapsed(i)}
                  handleFinishRunningPipeline={handleFinishRunningPipeline}
                  boxColor={boxColors[i]}
                  setBoxColor={(color) => handleSetBoxColor(i, color)}
                  number={i}
                />
              );
            } else if (step.type === 'Retrieval') {
              return (
                <RetrievalBox
                  key={i}
                  number={i}
                  query={step.query}
                  vanillaChunks={step.vanillaChunks}
                  raptorChunks={step.raptorChunks}
                  searchBy={step.searchBy}
                  retrievalModeIn={step.retrievalMode}
                  chunkSize={step.chunkSize}
                  chunkOverlap={step.chunkOverlap}
                  selectedChunks={step.selectedChunks}
                  k={step.k}
                  setIsCollapsed={() => handleSetCollapsed(i)}
                  isCollapsed={isCollapsed(i)}
                  id={step.id}
                  order={step.order}
                  handleFinishRunningPipeline={handleFinishRunningPipeline}
                  boxColor={boxColors[i]}
                  setBoxColor={(color) => handleSetBoxColor(i, color)}
                />
              );
            } else {
              return null;
            }
          })}
          {loading && (
            <LoadingBox
              title="🔎 loading... 🧠"
              options={[]}
              isCollapsed={true}
            />
          )}
        </div>

      </div>

    </>
  );
}

export default App;
