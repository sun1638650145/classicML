//
//  exceptions.h
//  exceptions
//
//  Created by 孙瑞琦 on 2021/1/21.
//

#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

namespace exceptions {
class NotImplementedError : public std::exception {
    public:
        const char *what() const noexcept override {
            return "函数没有实现";
        }
};
}  // namespace exceptions

#endif /* EXCEPTIONS_H */