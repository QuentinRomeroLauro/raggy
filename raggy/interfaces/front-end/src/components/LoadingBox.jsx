import './LoadingBox.css'; // Assuming you have some CSS for styling
import { AiOutlineLoading } from "react-icons/ai";
import PillButton from './PillButton';

const LoadingBox = ({ title, children, options, isCollapsed=true }) => {

    return (
        <div className={`collapsible-box ${isCollapsed ? 'collapsed' : 'expanded'}`}>
            <div className="header">
                <div className="title-and-options">
                    <span className="title">{title}</span>
                    <div className="options">
                        {options.map((option, index) => (
                            <PillButton text={option.label} onClick={option.onClick}/>
                            // <button key={index} onClick={option.onClick}>{option.label}</button>
                        ))}
                    </div>
                </div>
                <AiOutlineLoading size={32} className="icon"/>
            </div>

            {!isCollapsed && <div className="content"></div>}
        </div>
    );
};

export default LoadingBox;
