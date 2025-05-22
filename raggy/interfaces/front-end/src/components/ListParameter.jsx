import React from 'react';
import './ListParameter.css';


export default function ListParameter({ 
    parameterList, 
    title,
    startingOption,
    selectedParameter,
    setSelectedParameter,
}) {
    return (
        <div className="parameter" style={{fontSize: '20px'}}>
            <span className="parameter-name">{title}</span>
            <select value={selectedParameter} onChange={(e) => setSelectedParameter(e.target.value)}>
                {parameterList.map((parameter, index) => (
                    <option key={index} value={parameter}>{parameter}</option>
                ))}
            </select>
        </div>
    );
}