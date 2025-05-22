import CollapsibleBox from "./CollapsibleBox";
import DoubleEditableBox from "./DoubleEditableBox";
import { useState } from "react";
import { toast, Bounce } from 'react-toastify';
import CircularProgress from '@mui/material/CircularProgress'; // Make sure you have @mui/material installed


export default function LLMBox({
    promptText,
    responseText,
    setIsCollapsed = () => { },
    isCollapsed = true,
    handleFinishRunningPipeline = () => { },
    temperature = 0.7,
    maxTokens = 100,
    id,
    order,
    boxColor,
    setBoxColor,
    number,
}) {
    const [prompt, setPrompt] = useState(promptText);
    const [result, setResult] = useState(responseText);
    const [loading, setLoading] = useState(false);

    const handleRunStepLLM = async () => {
        console.log('running llm step with prompt: ' + prompt);
        setLoading(true);

        const response = await fetch('http://localhost:5001/get_whatIf_llm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                max_tokens: maxTokens,
                temperature: temperature,
            }),
        });

        if (response.ok) {
            console.log('llm step ran successfully');
            const data = await response.json();
            setResult(data.responseText);
            console.log("result", result);
        }

        setBoxColor('#fcb3b3');
        setLoading(false);
    }

    const data = {
        type: 'LLM',
        promptText: prompt,
        responseText: result,
        temperature: temperature,
        maxTokens: maxTokens,
        id: id,
        order: order,
    }

    const handleSaveTrace = () => {
        console.log('saving retrieval as golden trace');
        // send the data to the server
        fetch('http://localhost:5001/save_trace', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
    }

    // const handleGetCode = async () => {
    //     /*
    //     To minimize friction in going between code and interface we provide
    //     a way to generate pipeline code from the user interface.

    //     Here is what generated code might look like:
    //     ```
    //     response = llm(
    //         prompt=prompt,
    //         max_tokens=100,
    //         temperature=0.7,
    //     )
    //     ```
    //     */

    //     // don't change the formatting of this string literal or else the copied indentation will be off
    //     const code = `prompt="""${prompt.replace(/"/g, '\\"')}"""
    // response = llm(
    //     prompt=prompt,
    //     max_tokens=${maxTokens},
    //     temperature=${temperature},
    // )`;

    //     // Copy the code to the clipboard
    //     await navigator.clipboard.writeText(code);

    //     toast('code copied 📋', {
    //         position: "top-left",
    //         autoClose: 750,
    //         hideProgressBar: true,
    //         closeOnClick: true,
    //         pauseOnHover: false,
    //         draggable: false,
    //         progress: undefined,
    //         theme: "colored",
    //         transition: Bounce,
    //         });
    // }

    const handleCopyPrompt = async () => {
        const promptCopy = `prompt="""${prompt.replace(/"/g, '\\"')}"""`;

        await navigator.clipboard.writeText(promptCopy);
        toast('prompt copied 📋', {
            position: "top-left",
            autoClose: 750,
            hideProgressBar: true,
            closeOnClick: true,
            pauseOnHover: false,
            draggable: false,
            progress: undefined,
            theme: "colored",
            transition: Bounce,
        });
    }


    return (
        <CollapsibleBox
            title="llm 🧠"
            options={[
                { label: 'run all', onClick: () => handleFinishRunningPipeline(data, number) },
                { label: 'run step', onClick: handleRunStepLLM },
                { label: 'copy prompt', onClick: handleCopyPrompt },
                // { label: 'Option 2', onClick: () => console.log('Option 2 clicked') },
            ]}
            isCollapsed={isCollapsed}
            setIsCollapsed={setIsCollapsed}
            color={boxColor}
        >
            <div style={{ position: "relative" }}>

                <DoubleEditableBox
                    leftTitle="Prompt"
                    leftText={prompt}
                    setTextLeft={setPrompt}
                    rightTitle="Response"
                    rightText={result}
                    setTextRight={setResult}
                    editableLeft={false}
                    editableRight={false}
                    optionsLeft={[
                        // {label: 'copy prompt', onClick: handleCopyPrompt} 
                    ]}
                    optionsRight={[
                        // { label: 'save output as trace', onClick: handleSaveTrace }
                    ]}
                />
                
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