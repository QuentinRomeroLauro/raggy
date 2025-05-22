import React, { useState } from 'react';
import './CollapsibleBox.css'; // Assuming you have some CSS for styling
import { FaArrowLeft } from "react-icons/fa";
import { FaArrowDown } from "react-icons/fa";
import PillButton from './PillButton';

const CollapsibleBox = ({ title, 
    children, 
    options, 
    isCollapsed, 
    setIsCollapsed =() => {}, color='f0f0f0',
    semanticSimilarity,
 }) => {

    const toggleCollapse = () => {
        setIsCollapsed(!isCollapsed);
    };

    const evaluateColor = (semanticSimilarity) => {
        if (semanticSimilarity === undefined) {
            return '#f0f0f0'; // grey
        }
        if (semanticSimilarity > 0.7) {
            return '#77dd77'; // green
        } else if (semanticSimilarity > 0.3) {
            return '#fdfd96'; // yellow
        } else {
            return '#ff6961'; // red 
        }
    }


    return (
        <div 
            className={`collapsible-box ${isCollapsed ? 'collapsed' : 'expanded'}`}
            style={{backgroundColor: color}}
        >
            

            <div className="header" onClick={toggleCollapse}>
                <div className="title-and-options">
                    <span className="title">{title}</span>
                    <div className="options">
                        {options.map((option, index) => (
                            <PillButton key={index} text={option.label} onClick={option.onClick}/>
                        ))}
                    </div>
                    <div className="score">
                        {semanticSimilarity && <PillButton text={'similarity: ' + semanticSimilarity.toFixed(2)} onClick={() => {}} color={evaluateColor(semanticSimilarity)} />}
                    </div>
                </div>
                
                {isCollapsed ? <FaArrowLeft className="collapse-icon" /> : <FaArrowDown  className="collapse-icon"/>}
            </div>
            
            {!isCollapsed && <div className="content">{children}</div>}
        </div>
    );
};

export default CollapsibleBox;
