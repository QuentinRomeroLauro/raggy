import { useEffect, useState } from 'react';
import './ChunkBox.css';
import PillButton from './PillButton';
import Chunk from './Chunk';
import SearchBar from './SearchBar';

const ChunkBox = ({ title = "chunks 🪵",
    children,
    options = [],
    searchOptions = ["option 1", "option 2"],
    currChunks,
    toggleChunkSelection,
    currSelectedChunks,
    chunkSize,
    chunkOverlap,
    k,
    retrievalMode,
    setCurrChunks,
}) => {

    const [searchTerm, setSearchTerm] = useState('');
    const [selectedOption, setSelectedOption] = useState(searchOptions[0]);


    // call the end point to get the chunks based off the current params, but with this custom query
    const handleSearch = async () => {

        // call the what if endpoint
        const response = await fetch('http://localhost:5001/get_whatIf_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: searchTerm,
                searchBy: selectedOption,
                chunkSize: chunkSize,
                chunkOverlap: chunkOverlap,
                k: k,
                retrievalMode: retrievalMode,
            })
        });

        if (response.ok) {
            const data = await response.json();
            console.log(data)
            if (retrievalMode === 'raptor') {
                setCurrChunks(data.raptorChunks);
            } else if(retrievalMode === 'vanilla'){
                setCurrChunks(data.vanillaChunks);
            } else {
                throw new Error('Invalid retrieval mode');
            }
        }
    }

    const isChunkSelected = (chunk) => {
        for(let i = 0; i < currSelectedChunks.length; i++){
            if(currSelectedChunks[i].id === chunk.id && currSelectedChunks[i].text === chunk.text){
                return true;
            }
        }
        return false;
    }



    return (
        <div className="chunk-box">
            <div className="chunk-box-header">
                <span className="innerTitle" style={{fontSize: '20px'}}>{title}</span>
                <div className="chunk-button-container">
                    {options?.map((option, index) => (
                        <PillButton
                            key={index}
                            text={option.label}
                            onClick={option.onClick}
                        />
                    ))
                    }

                </div>
            </div>
            <div>
                {/* Search Bar */}
                <SearchBar
                    placeholder="Search 🔍"
                    options={searchOptions}
                    searchTerm={searchTerm}
                    setSearchTerm={setSearchTerm}
                    handleSearch={handleSearch}
                    selectedOption={selectedOption}
                    setSelectedOption={setSelectedOption}
                />
            </div>
            <div className="parameters">
                {/* All the different edit options */}
                {currChunks?.map((chunk, index) => (
                    <Chunk
                        key={index}
                        title={"Chunk " + chunk.id}
                        text={chunk.text}
                        isSelected={isChunkSelected(chunk)}
                        score={chunk.score}
                        onToggleSelect={() => toggleChunkSelection(chunk)}
                    />
                ))}
            </div>

        </div>
    );
};

export default ChunkBox;
