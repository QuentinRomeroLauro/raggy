import CollapsibleBox from "./CollapsibleBox";
import { EditableBox } from "./DoubleEditableBox";

export default function QueryBox({
    query,
    setQuery,
    id,
    order,
    setIsCollapsed = () => { },
    isCollapsed = true,
    handleFinishRunningPipeline = () => { },
    number,
}) {

    const data = {
        type: 'Query',
        query: query,
        id: id,
        order: order,
    }

    return (
        <CollapsibleBox
            title="query ❓"
            options={[
                { label: 'run pipeline', onClick: () => handleFinishRunningPipeline(data, number) },
            ]}
            isCollapsed={isCollapsed}
            setIsCollapsed={setIsCollapsed}
        >
            <EditableBox
                title="query ❓"
                isEditable={false}
                text={query}
                topRightButtonText={"edit query"}
                setText={setQuery}
            />

        </CollapsibleBox>
    );
} 