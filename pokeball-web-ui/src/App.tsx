import {useCallback, useState} from 'react'
import './App.scss'
import {selectFiles} from "sz-react-support"
import {getBase64} from "./utils"
import icon from "./assets/icon.png"
import {APIResult} from "./APIResult.ts";

interface Task {
    file: File
    base64: string
    status: "prepared" | "reasoned"
    label: string | null
}

function App() {
    const [tasks, setTasks] = useState<Task[]>([])
    const uploadFiles = useCallback(async () => {
        const _files = [...await selectFiles({
            multiple: true
        })]
        const addition: Task[] = []
        for (const f of _files) {
            addition.push({
                file: f,
                base64: await getBase64(f),
                status: "prepared",
                label: null
            })
        }
        setTasks([...tasks, ...addition])
    }, [tasks])

    const doReasoning = useCallback(async () => {
        const form = new FormData()
        for (const task of tasks) {
            form.append("files", task.file)
        }
        const resp = await fetch("http://localhost:22401", {
            method: "POST",
            body: form,
        })
        const data: APIResult = await resp.json()
        if (data.success) {
            for (let i = 0; i < data.data.length; i++) {
                tasks[i].status = "reasoned"
                tasks[i].label = `${data.data[i][0][0]} (${(data.data[i][0][1] * 100).toFixed(2)}%)`
            }
            setTasks([...tasks])
        }


    }, [tasks])

    return (
        <>
            <img style={{borderRadius: "30px", padding: "20px", background: "white"}} src={icon}></img>
            <h1>Poké Ball</h1>
            <p>Pokemon Classification based on deep learning</p>
            <h2 style={{fontWeight: "lighter"}}>A SEYMOUR ZHANG A.I.</h2>
            <button onClick={doReasoning}>立刻执行推理 {tasks.length}</button>
            <button onClick={uploadFiles}>上传文件</button>
            <button onClick={() => setTasks([])}>清空</button>
            <div className='container'>
                {
                    tasks.map(task => {
                        return <div className="card">
                            <img src={task.base64}/>
                            <div>
                                <p>
                                    {
                                        task.status === "prepared" ? "未推理" : "已推理"
                                    }
                                </p>
                                {
                                    task.status === "reasoned" &&
                                    <p>这是：{task.label}</p>
                                }
                            </div>
                        </div>
                    })
                }
            </div>

        </>
    )
}

export default App
