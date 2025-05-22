import React, { useRef } from 'react';
import './Chunk.css';

// Chunk Interface:
// {
//     text: "chunk text",
//     score: {chunk score }
//     rank: {chunk rank }
//     chunk_id: {chunk id }
// }

const Chunk = ({
    title,
    isSelected,
    score,
    text,
    onToggleSelect,
}) => {
    const chunkRef = useRef(null);

    const handleClick = (e) => {
        if (chunkRef.current) {
            const rect = chunkRef.current.getBoundingClientRect();
            const isInBounds = (
                e.clientX >= rect.left &&
                e.clientX <= rect.right &&
                e.clientY >= rect.top &&
                e.clientY <= rect.bottom
            );

            if (isInBounds) {
                onToggleSelect();
            }
        }
    };


    return (
        <div onClick={handleClick} ref={chunkRef} className={`chunk ${isSelected ? 'selected' : ''}`}>
            <div className="chunk-header" style={{fontSize: '15px'}}>
                <span>{title}</span>

                <p className="toggle-select">
                    {isSelected ? 'unselect chunk 👇' : 'select chunk 👇'}
                </p>

                <span>score: {(1 - Math.pow((1-score), 2)).toFixed(2)}</span>
            </div>
            <div className="chunk-text" style={{fontSize: '15px'}}>
                <div style={{
                    fontFamily: "'Noto Sans', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'Noto Sans CJK JP', Arial, sans-serif",
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    overflowWrap: 'break-word'
                }}>
                    {text}
                </div>
            </div>
        </div>
    );
};

export default Chunk;
