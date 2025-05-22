import { EditableBox } from "./DoubleEditableBox";
import CollapsibleBox from "./CollapsibleBox";

export default function AnswerBox({
    answer,
    setAnswer,
    id,
    order,
    setIsCollapsed = () => { },
    isCollapsed = true,
    handleSaveTrace = () => { },
    boxColor,
    setBoxColor,
    semanticSimilarity,
}) {

    const data = {
        type: 'Answer',
        answer: answer,
        id: id,
        order: order,
    }

    return (
        <CollapsibleBox
            title="answer ✅"
            options={[
                { label: 'save answer', onClick: handleSaveTrace },
            ]}
            isCollapsed={isCollapsed}
            setIsCollapsed={setIsCollapsed}
            color={boxColor}
            semanticSimilarity={semanticSimilarity}
        >
            <EditableBox
                title="answer ✅"
                isEditable={true}
                text={answer}
                topRightButtonText={"edit final answer"}
                setText={setAnswer}
            />

        </CollapsibleBox>
    );
} 