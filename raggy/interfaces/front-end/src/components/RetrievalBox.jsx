import React, { useState } from 'react';
import './RetrievalBox.css';
import { toast, Bounce } from 'react-toastify';
import ParameterBox from './ParameterBox';
import TextParameter from './TextParameter';
import ListParameter from './ListParameter';
import ChunkBox from './ChunkBox';
import Chunk from './Chunk';
import CollapsibleBox from './CollapsibleBox';
import CircularProgress from '@mui/material/CircularProgress'; // Make sure you have @mui/material installed

// import { exampleChunks } from './ExampleChunks';
import { EditableBox } from './DoubleEditableBox';
// import io from 'socket.io-client';
import ChunkChart from './ChunkChart';

export default function RetrievalBox({
    toggleChunk = () => { },
    query = 'query',
    id,
    order,
    selectedChunks = [],
    vanillaChunks = [],
    raptorChunks = [],
    retrievalModeIn = 'vanilla',
    searchBy = 'semantic',
    chunkSize = 100,
    chunkOverlap = 50,
    k = 5,
    setIsCollapsed = () => { },
    isCollapsed = true,
    handleFinishRunningPipeline = () => { },
    boxColor,
    setBoxColor,
    number,
}) {

    const searchOptions = ['semantic similarity', 'max marginal relevance', 'tfidf'];

    const [loading, setLoading] = useState(false);

    // query
    const [currQuery, setCurrQuery] = useState(query);

    // chunks
    const [currVanillaChunks, setCurrVanillaChunks] = useState(vanillaChunks);
    const [currRaptorChunks, setCurrRaptorChunks] = useState(raptorChunks);
    const [currSelectedChunks, setCurrSelectedChunks] = useState(selectedChunks);
    console.log('selected chunks:', currSelectedChunks);

    // parameters
    const [currSearchBy, setCurrSearchBy] = useState(searchBy);
    const [currChunkSize, setCurrChunkSize] = useState(chunkSize);
    const [currChunkOverlap, setCurrChunkOverlap] = useState(chunkOverlap);
    const [currK, setCurrK] = useState(k);

    // retrieval mode
    const [retrievalMode, setRetrievalMode] = useState(retrievalModeIn); // vanilla || raptor


    const paramChanged = currSearchBy !== searchBy || currChunkSize !== chunkSize || currChunkOverlap !== chunkOverlap || currK !== k;
    const queryChanged = currQuery !== query;

    const toggleChunkSelection = (chunk) => {
        console.log('toggling chunk selection:', chunk.id);
        currVanillaChunks?.forEach((chunk_i) => {
            if (chunk.text === chunk_i.text) {
                const isChunkSelected = currSelectedChunks.some(selectedChunk =>
                    selectedChunk.id.toString() === chunk.id.toString() &&
                    selectedChunk.text.toString() === chunk.text.toString() &&
                    selectedChunk.score.toString() === chunk.score.toString()
                );
            
                if (isChunkSelected) {
                    setCurrSelectedChunks(currSelectedChunks.filter((selectedChunk) =>
                        !(selectedChunk.text.toString() === chunk_i.text.toString() &&
                          selectedChunk.score.toString() === chunk_i.score.toString() &&
                          selectedChunk.id.toString() === chunk_i.id.toString())
                    ));
                } else {
                    setCurrSelectedChunks([...currSelectedChunks, chunk_i]);
                }
            }
        });
        currRaptorChunks?.forEach((chunk_i) => {
            if (chunk.text === chunk_i.text) {
                const isChunkSelected = currSelectedChunks.some(selectedChunk =>
                    selectedChunk.id.toString() === chunk.id.toString() &&
                    selectedChunk.text.toString() === chunk.text.toString() &&
                    selectedChunk.score.toString() === chunk.score.toString()
                );
        
                if (isChunkSelected) {
                    setCurrSelectedChunks(currSelectedChunks.filter((selectedChunk) =>
                        !(selectedChunk.text.toString() === chunk_i.text.toString() &&
                          selectedChunk.score.toString() === chunk_i.score.toString() &&
                          selectedChunk.id.toString() === chunk_i.id.toString())
                    ));
                } else {
                    setCurrSelectedChunks([...currSelectedChunks, chunk_i]);
                }
            }
        });
    };

    const handleWhatIf = async () => {
        console.log("running what if with searchBy: " + searchBy + ", chunkSize: " + chunkSize + ", chunkOverlap: " + chunkOverlap + ", k: " + k + ", retrievalMode: " + retrievalMode);
        setLoading(true);
        const response = await fetch('http://localhost:5001/get_whatIf_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: currQuery,
                searchBy: currSearchBy,
                chunkSize: currChunkSize,
                chunkOverlap: currChunkOverlap,
                k: currK,
                retrievalMode: retrievalMode,
            })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('What If data:', data);
            // Update the states to reflect the whatif retrieval
            setCurrQuery(data.query);
            setCurrSearchBy(data.searchBy);
            setCurrChunkSize(data.chunkSize);
            setCurrChunkOverlap(data.chunkOverlap);
            setCurrK(data.k);
            setCurrVanillaChunks(data.vanillaChunks);
            setCurrRaptorChunks(data.raptorChunks);
            setRetrievalMode(data.retrievalMode);
            setCurrSelectedChunks(data.selectedChunks);
            setBoxColor('#fcb3b3');
        } else {
            console.error('Failed to run What If retrieval');
        }

        setLoading(false);
    };

    const handleGenerateCode = async () => {
        /*
        To minimize friction in going between code and interface we provide
        a way to generate pipeline code from the user interface.

        Here is what generated code might look like: 
        ```
        docs_and_scores = retriever.invoke(
                query=q,
                k=4,
                chunkSize=200,
                chunkOverlap=100,
            )
        ```
        */

        const code = `docs_and_scores = retriever.invoke(
            query=str(query),
            k=${currK},
            chunkSize=${currChunkSize},
            chunkOverlap=${currChunkOverlap},
        )`;

        // Copy the code to the clipboard
        await navigator.clipboard.writeText(code);

        toast('code copied 📋', {
            position: "top-left",
            autoClose: 750,
            hideProgressBar: true,
            closeOnClick: true,
            pauseOnHover: false,
            draggable: false,
            progress: undefined,
            theme: "dark",
            transition: Bounce,
        });
        console.log("here")
    }

    const handleSaveTrace = () => {
        console.log('saving retrieval as golden trace');
        // send the data to the server
        fetch('http://localhost:5001/save_trace', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify( {
                type: 'Retrieval',
                query: currQuery,
                searchBy: currSearchBy,
                chunkSize: currChunkSize,
                chunkOverlap: currChunkOverlap,
                k: currK,
                vanillaChunks: currVanillaChunks,
                raptorChunks: currRaptorChunks,
                selectedChunks: currSelectedChunks,
                retrievalMode: retrievalMode,
                id: id,
                order: order,
            }),
        });
    }

    return (

        <CollapsibleBox
            title="retriever 🔍"
            options={[
                // { label: 'fork', onClick: () => console.log('fork clicked') },
                { label: 'run all', onClick: () => handleFinishRunningPipeline({
                    type: 'Retrieval',
                    query: currQuery,
                    searchBy: currSearchBy,
                    chunkSize: currChunkSize,
                    chunkOverlap: currChunkOverlap,
                    k: currK,
                    vanillaChunks: currVanillaChunks,
                    raptorChunks: currRaptorChunks,
                    selectedChunks: currSelectedChunks,
                    retrievalMode: retrievalMode,
                    id: id,
                    order: order,
                }, number) },
                { label: 'run step', onClick: handleWhatIf },
                { label: 'copy code', onClick: handleGenerateCode },
                // { label: 'view chunk chart', onClick: () => console.log('view chunk chart clicked') },
                // { label: 'Option 2', onClick: () => console.log('Option 2 clicked') },
            ]}
            isCollapsed={isCollapsed}
            setIsCollapsed={setIsCollapsed}
            color={boxColor}
        >

            <div style={{position: "relative"}}>
                <div className="retrieval-box">
                    <div className="query-and-params">
                        <EditableBox
                            title="query"
                            isEditable={false}
                            text={currQuery}
                            setText={setCurrQuery}
                            topRightButtonText="edit"
                            backgroundColor={queryChanged ? '#f0f0f0' : '#ffffff'}
                        />

                        <ParameterBox
                            options={[
                                // { label: 'run what if❓', onClick: handleWhatIf},
                            ]}
                            backgroundColor={paramChanged ? '#f0f0f0' : '#ffffff'}
                        >
                            <ListParameter
                                title="search by"
                                parameterList={searchOptions}
                                selectedParameter={currSearchBy}
                                setSelectedParameter={setCurrSearchBy}
                            />
                            {/* <ListParameter
                                title="retrieval mode"
                                parameterList={["vanilla", "raptor"]}
                                selectedParameter={retrievalMode}
                                setSelectedParameter={setRetrievalMode}
                            /> */}
                            <ListParameter
                                title="chunk size"
                                parameterList={[100, 200, 400, 800, 1000, 1500, 2000]}
                                selectedParameter={currChunkSize}
                                setSelectedParameter={setCurrChunkSize}
                            />
                            <ListParameter
                                title="chunk overlap"
                                parameterList={[0, 10, 25, 50, 100, 200, 400]}
                                selectedParameter={currChunkOverlap}
                                setSelectedParameter={setCurrChunkOverlap}
                            />
                            <TextParameter
                                title="k"
                                text={currK}
                                setText={setCurrK}
                            />

                        </ParameterBox>

                    </div>

                    <div className="chunk-holder">
                        {/* VANILLA CHUNKS */}
                        <ChunkBox 
                            title={'chunks 🪵'}
                            options={[
                                // { label: 'save selected chunks as trace', onClick: handleSaveTrace },
                            ]}
                            searchOptions={searchOptions}
                            className="chunk-box"
                            toggleChunkSelection={toggleChunkSelection}
                            currSelectedChunks={currSelectedChunks}
                            currChunks={currVanillaChunks}
                            chunkSize={currChunkSize}
                            chunkOverlap={currChunkOverlap}
                            k={currK}
                            retrievalMode={retrievalMode}
                            setCurrChunks={setCurrVanillaChunks}
                        />
                    </div>

                    {/* Commenting out RAPTOR Chunks b/c it just confused the user */}
                     {/* <div> */}
                        {/* CLUSTER CHUNKS */}
                        {/* <ChunkBox */}
                            {/* title={'raptor summaries 🦖'}
                            options={[
                                // { label: 'save selected chunks as trace', onClick: handleSaveTrace },
                            ]}
                            searchOptions={searchOptions}
                            toggleChunkSelection={toggleChunkSelection}
                            currSelectedChunks={currSelectedChunks}
                            className="chunk-box"
                            currChunks={currRaptorChunks}
                            chunkSize={currChunkSize}
                            chunkOverlap={currChunkOverlap}
                            k={currK}
                            retrievalMode={retrievalMode}
                            setCurrChunks={setCurrRaptorChunks} */}
                        {/* /> */}
                    {/* </div>  */}

                </div>


                <div>
                    <ChunkChart 
                        vanillaChunks={currVanillaChunks}
                        // raptorChunks={[currRaptorChunks]}
                        raptorChunks={[]} // to hide the raptor chunk visualization
                        chunks={currVanillaChunks} selectedChunks={currSelectedChunks} />
                </div>

                {loading && ( 
                    <div style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        backgroundColor: 'rgba(0, 0, 0, 0.1)',
                        borderRadius: '20px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        zIndex: 10,
                    }}>
                        <CircularProgress 
                            size={64} 
                            thickness={4} 
                            color="secondary" 
                            style={{color: 'white'}}
                        />
                    </div>
                )}


            </div>

        </CollapsibleBox>
    );
}