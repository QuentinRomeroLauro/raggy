import React, { useState } from 'react';
import './DoubleEditableBox.css';
import PillButton from './PillButton';

export const EditableBox = ({ title, isEditable=true, text, setText=() => {}, topRightButtonText, isResponse, backgroundColor="#ffffff", options=[] }) => {
    const [editable, setEditable] = useState(isEditable);

    const handleContentChange = (e) => {
        if (editable) {
            setText(e.target.value);
        }
    };

    const toggleEditability = () => {
        setEditable(!editable);
    };

    return (
        <div className="editable-box" style={{backgroundColor: backgroundColor}}>
            <div className="box-header">
                <span className="innerTitle" style={{fontSize: '20px'}}>{title}</span>
                <div className="top-buttons">

                    {/* Other options */}
                    {options.map((option, index) => (
                        <PillButton text={option.label} onClick={option.onClick}/>
                        // <button key={index} onClick={option.onClick}>{option.label}</button>
                    ))}

                    {/* Editable Toggle */}
                    <PillButton 
                        onClick={toggleEditability} 
                        // className="edit-button"
                        text={editable ? 'save changes': 'edit'}
                    /> 
                </div>
            </div>
            <textarea
                value={text}
                onChange={handleContentChange}
                readOnly={!editable}
                className="editContent"
                style={{fontSize: '30px'}}
            />
        </div>
    );
};

const DoubleEditableBox = ({leftTitle, 
                            rightTitle, 
                            editableLeft, 
                            editableRight, 
                            leftText, 
                            rightText, 
                            setTextLeft=() => {},
                            setTextRight=() => {},
                            topRightButtonTextLeft,
                            topRightButtonTextRight,
                            optionsLeft,
                            optionsRight,
                        }) => {
    return (
        <div className="double-editable-box">
            <EditableBox 
                title={leftTitle}
                isEditable={true} 
                text={leftText} 
                topRightButtonText={"edit prompt"}
                setText={setTextLeft}
                options={optionsLeft}
            />
            <EditableBox 
                title={rightTitle} 
                isEditable={editableRight} 
                text={rightText} 
                topRightButtonText={"edit response"}
                isResponse={true}
                setText={setTextRight}
                options={optionsRight}
            />
        </div>
    );
};

export default DoubleEditableBox;
