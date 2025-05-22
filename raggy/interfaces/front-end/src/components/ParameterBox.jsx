import React from 'react';
import './ParameterBox.css';
import PillButton from './PillButton';

const ParameterBox = ({ title="parameters", children, options=[], backgroundColor="#ffffff" }) => {
    
    return (
        <div className="parameter-box" style={{backgroundColor: backgroundColor}}>
            <div className="parameter-box-header">
                <span className="innerTitle" style={{fontSize: '20px'}}>{title}</span>
            </div>
            <div className="parameters" style={{fontSize: '20px'}}>
                {/* All the different edit options */}
                {children}
            </div>
            <div className="button-container">
                {options.map((option, index) => (
                    <PillButton 
                        key={index}
                        text={option.label}
                        onClick={option.onClick}
                    />
                ))    
                }
            </div>
        </div>
    );
};

export default ParameterBox;
